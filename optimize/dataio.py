import h5py
import numpy as np
from numpy.lib import recfunctions as rfn
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        self.index = 0
        return self

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration

        # Ensure batch_indices is a NumPy array of integers
        batch_indices = np.array(self.indices[self.index:self.index + self.batch_size], dtype=int)

        # Directly index into the dataset using batch_indices
        batch = [self.dataset[i] for i in batch_indices]

        # Keep the batch as a host NumPy array (consistent boundary)
        batch = np.array(batch)

        self.index += self.batch_size
        return batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def np_from_structured(tracks):
    # structured_to_unstructured returns a NumPy array; keep it as NumPy
    tracks_np = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)
    return np.asarray(tracks_np, dtype=np.float32)

def remap_event_ids_to_local(batch_arr, source_event_ids, batch_event_ids, fields):
    evt_col = fields.index("eventID")
    source_event_ids = np.asarray(source_event_ids, dtype=np.int64)
    batch_event_ids = np.asarray(batch_event_ids, dtype=np.int64)

    if source_event_ids.size == 0:
        return batch_arr

    local_event_ids = np.searchsorted(batch_event_ids, source_event_ids)
    if np.any(local_event_ids >= batch_event_ids.size) or not np.array_equal(batch_event_ids[local_event_ids], source_event_ids):
        missing = np.setdiff1d(np.unique(source_event_ids), batch_event_ids, assume_unique=False)
        raise ValueError(f"Failed to remap eventID to local indices. Missing global eventID values: {missing[:16].tolist()}")

    batch_arr[:, evt_col] = local_event_ids.astype(np.float32)
    return batch_arr

def chop_tracks(tracks, fields, precision=0.001):
    def split_track(track, nsteps, length, direction, i):
        new_tracks = track.reshape(1, track.shape[0]).repeat(nsteps, axis=0)

        new_tracks[:, fields.index("dE")] = new_tracks[:, fields.index("dE")]*precision/(length+1e-10)
        steps = np.arange(0, nsteps)

        new_tracks[:, fields.index("x_start")] = track[fields.index("x_start")] + steps*precision*direction[0]
        new_tracks[:, fields.index("y_start")] = track[fields.index("y_start")] + steps*precision*direction[1]
        new_tracks[:, fields.index("z_start")] = track[fields.index("z_start")] + steps*precision*direction[2]

        new_tracks[:, fields.index("x_end")] = track[fields.index("x_start")] + precision*(steps + 1)*direction[0]
        new_tracks[:, fields.index("y_end")] = track[fields.index("y_start")] + precision*(steps + 1)*direction[1]
        new_tracks[:, fields.index("z_end")] = track[fields.index("z_start")] + precision*(steps + 1)*direction[2]
        new_tracks[:, fields.index("dx")] = precision

        #Correcting the last track bit
        new_tracks[-1, fields.index("x_end")] = track[fields.index("x_end")]
        new_tracks[-1, fields.index("y_end")] = track[fields.index("y_end")]
        new_tracks[-1, fields.index("z_end")] = track[fields.index("z_end")]
        new_tracks[-1, fields.index("dE")] = track[fields.index("dE")]*(1 - precision*(nsteps - 1)/(length + 1e-10))
        new_tracks[-1, fields.index("dx")] = length - precision*(nsteps - 1)

        #Finally computing the middle point once everything is ok
        new_tracks[:, fields.index("x")] = 0.5*(new_tracks[:, fields.index("x_start")] + new_tracks[:, fields.index("x_end")])
        new_tracks[:, fields.index("y")] = 0.5*(new_tracks[:, fields.index("y_start")] + new_tracks[:, fields.index("y_end")])
        new_tracks[:, fields.index("z")] = 0.5*(new_tracks[:, fields.index("z_start")] + new_tracks[:, fields.index("z_end")])

        return new_tracks
    
    start = np.stack([tracks[:, fields.index("x_start")],
                        tracks[:, fields.index("y_start")],
                        tracks[:, fields.index("z_start")]], axis=1)
    end = np.stack([tracks[:, fields.index("x_end")],
                    tracks[:, fields.index("y_end")],
                    tracks[:, fields.index("z_end")]], axis=1)

    segment = end - start
    length = np.sqrt(np.sum(segment**2, axis=1))
    eps = 1e-10
    direction = segment / (length[:, None] + eps)
    nsteps = np.maximum(np.ceil(length / precision), 1).astype(int).flatten()
    new_tracks = np.vstack([split_track(tracks[i], nsteps[i], length[i], direction[i], i) for i in range(tracks.shape[0])])
    return new_tracks

class TracksDataset:
    def __init__(self, filename, nevents, max_nbatch=None, swap_xz=True, random_nevents=False, data_seed=42, track_len_sel=2., 
                 max_abs_costheta_sel=0.966, min_abs_segz_sel=15., track_z_bound=28., max_batch_len=50, print_input=False,
                 chopped=True, pad=True, electron_sampling_resolution=0.1, live_selection=False):

        # Build per-batch mappings so __getitem__ can construct a single batch on demand.
        with h5py.File(filename, 'r') as f:
            tracks = f['segments'][:] # convert to array

        if swap_xz:
            x_start = np.copy(tracks['x_start'] )
            x_end = np.copy(tracks['x_end'])
            x = np.copy(tracks['x'])

            tracks['x_start'] = np.copy(tracks['z_start'])
            tracks['x_end'] = np.copy(tracks['z_end'])
            tracks['x'] = np.copy(tracks['z'])

            tracks['z_start'] = x_start
            tracks['z_end'] = x_end
            tracks['z'] = x

        if not 't0' in tracks.dtype.names:
            tracks = rfn.append_fields(tracks, 't0', np.zeros(tracks.shape[0]), usemask=False)
        
        self.track_fields = tracks.dtype.names

        replace_map = {
            'event_id': 'eventID',
            'traj_id': 'trackID',
        }
        self.track_fields = tuple([replace_map.get(field, field) for field in self.track_fields])

        tracks.dtype.names = self.track_fields

        # Only load useful tracks
        # assuming tracks are in orders as a unit of trajectory
        # Trim at event boundaries so no event is split during initial load.
        if max_batch_len is not None and max_nbatch is not None and max_nbatch > 0:
            length_load_threshold = max_batch_len * (max_nbatch + 2)

            cum_dx = np.cumsum(tracks['dx'])
            cutoff = int(np.searchsorted(cum_dx, length_load_threshold, side='left'))
            if cutoff < len(tracks):
                cutoff_event_id = tracks['eventID'][cutoff]
                # Find the last row belonging to a *different* event before the cutoff
                mask = tracks['eventID'][:cutoff + 1] != cutoff_event_id
                if mask.any():
                    last_complete_row = int(np.where(mask)[0][-1])
                    tracks = tracks[:last_complete_row + 1]
        
        ##############################
        # Build traj-seg 2D mapping
        ##############################
        if live_selection:
            selected_tracks = tracks[abs(tracks['z']) < min_abs_segz_sel]

            unique_tracks, first_indices = np.unique(selected_tracks[['eventID', 'trackID']], return_index=True)

            first_indices = np.sort(first_indices)
            last_indices = np.r_[first_indices[1:] - 1, len(selected_tracks) - 1]
            n_repeat = last_indices - first_indices + 1

            tracks_start = selected_tracks[first_indices]
            tracks_end = selected_tracks[last_indices]
            tracks_xd = tracks_start['x_start'] - tracks_end['x_end']
            tracks_yd = tracks_start['y_start'] - tracks_end['y_end']
            tracks_zd = tracks_start['z_start'] - tracks_end['z_end']

            tracks_dir = np.column_stack((tracks_xd, tracks_yd, tracks_zd))
            z_dir = np.array([0, 0, 1])
            cos_theta = np.abs(np.dot(tracks_dir, z_dir))/ (np.linalg.norm(tracks_dir, axis=1) + 1e-10)

            trk_mask = np.sqrt(tracks_xd**2 + tracks_yd**2 + tracks_zd**2) > track_len_sel
            trk_mask = trk_mask & (cos_theta < max_abs_costheta_sel)
            trk_mask = trk_mask & (np.maximum(abs(tracks_start['z']), abs(tracks_end['z'])) < track_z_bound)
            mask = np.repeat(trk_mask, n_repeat)

            keys = np.ascontiguousarray(selected_tracks[mask][['eventID', 'trackID']])
        else:
            keys = np.ascontiguousarray(tracks[['eventID', 'trackID']])

        # Keep trajectory keys (eventID, trackID) aligned with trajectory ids.
        # Build per-row trajectory index mapping using unique keys for the selected rows
        # compute unique trajectory keys for the whole file and inverse mapping
        index, inverse_idx = np.unique(keys, return_inverse=True)
        self.traj_keys = np.ascontiguousarray(index)

        # Build per-trajectory list of row indices.
        # Assemble batches on-demand instead of storing full arrays.
        self.tracks_struct = tracks  # keep reference to structured rows on host
        nb_rows = tracks.shape[0]
        nb_trajs = len(index)

        # inverse_idx: 1D int array of length N_rows, values in [0, nb_trajs)
        sorted_idx = np.argsort(inverse_idx)          # indices of rows sorted by traj id (C)
        sorted_vals = inverse_idx[sorted_idx]         # sorted trajectory ids

        unique_vals, start_idx = np.unique(sorted_vals, return_index=True)
        end_idx = np.append(start_idx[1:], len(sorted_vals))

        # Preallocate list of empty arrays
        traj_row_indices = [np.empty(0, dtype=int) for _ in range(nb_trajs)]

        # Fill only for traj ids that actually occur
        for val, s, e in zip(unique_vals, start_idx, end_idx):
            traj_row_indices[int(val)] = sorted_idx[s:e]

        trajectory_row_indices = traj_row_indices # default order

        # Get eventID for the first row of each trajectory to identify which event it belongs to.
        event_ids = np.array([
            self.tracks_struct[trajectory_row_indices[i][0]]['eventID']
            for i in range(len(trajectory_row_indices))
        ])
        unique_events, first_event_idx = np.unique(event_ids, return_index=True)
        ordered_unique_events = unique_events[np.argsort(first_event_idx)]

        if nevents is not None and nevents > 0:
            if random_nevents:
                rng = np.random.default_rng(seed=data_seed)
                if nevents < len(ordered_unique_events):
                    selected_events = rng.choice(ordered_unique_events, size=nevents, replace=False)
                else:
                    selected_events = ordered_unique_events
            else:
                selected_events = ordered_unique_events[:nevents]

            selected_event_mask = np.isin(event_ids, selected_events)
            trajectory_row_indices = [
                trajectory_row_indices[idx]
                for idx in np.where(selected_event_mask)[0]
            ]

        ##################################
        self.trajectory_row_indices = trajectory_row_indices

        if max_batch_len is not None:
            # If not already computed, compute per-trajectory lengths
            traj_len = np.array([self.tracks_struct[rows]['dx'].sum() for rows in trajectory_row_indices])

            # Filter out trajectories longer than max_batch_len upfront
            valid_traj_mask = traj_len <= max_batch_len
            valid_traj_indices = np.where(valid_traj_mask)[0]

            lengths_ft = traj_len.copy()[valid_traj_mask]
            if lengths_ft.size == 0:
                raise ValueError("All tracks are longer than the batch size! Please check.")

            # Group valid trajectories by eventID so no event is split across batches.
            valid_event_ids = np.array([
                self.tracks_struct[trajectory_row_indices[vi][0]]['eventID']
                for vi in valid_traj_indices
            ])
            unique_evt_ids, evt_inv = np.unique(valid_event_ids, return_inverse=True)
            n_evts = len(unique_evt_ids)

            evt_traj_groups = [[] for _ in range(n_evts)]
            evt_lengths = np.zeros(n_evts)
            for pos, (evt_idx, length) in enumerate(zip(evt_inv, lengths_ft)):
                evt_traj_groups[evt_idx].append(valid_traj_indices[pos])
                evt_lengths[evt_idx] += length

            # Pack whole events into batches using the same floor-divide strategy.
            cumsum_evt_lengths = np.cumsum(evt_lengths)
            split_points = np.where(np.diff(np.floor_divide(cumsum_evt_lengths, max_batch_len)) > 0)[0] + 1
            split_points = np.append(split_points, len(cumsum_evt_lengths))

            split_points = np.insert(split_points, 0, 0)
            if max_nbatch and max_nbatch > 0:
                split_points = split_points[:(max_nbatch+1)]

            # Build batch -> trajectory index lists
            batches_traj_indices = []
            for i in range(len(split_points)-1):
                batch_traj = []
                for evt_idx in range(split_points[i], split_points[i+1]):
                    batch_traj.extend(evt_traj_groups[evt_idx])
                batches_traj_indices.append(batch_traj)

            self.batch_traj_indices = batches_traj_indices
            tot_data_length = cumsum_evt_lengths[split_points[-1]-1]

        else:
            # If no max_batch_len and no precomputed batch mapping, make one trajectory per batch
            if not hasattr(self, 'batch_traj_indices'):
                self.batch_traj_indices = [[i] for i in range(len(trajectory_row_indices))]
            # compute total track length as sum of dx across trajectories
            tot_data_length = np.sum([self.tracks_struct[rows]['dx'].sum() for rows in self.trajectory_row_indices])

        min_bt_idx = min(len(bt_idx) for bt_idx in self.batch_traj_indices)
        if min_bt_idx == 0:
            raise ValueError("There exist some empty batch in the simulation input!")

        self.batch_nsteps = []
        for batch_idxs in self.batch_traj_indices:
            rows_list = [self.trajectory_row_indices[t] for t in batch_idxs]
            if len(rows_list) == 0:
                self.batch_nsteps.append(0)
                continue

            rows = np.concatenate(rows_list)
            dx_vals = self.tracks_struct[rows]['dx']
            row_nsteps = np.maximum(np.ceil(dx_vals / electron_sampling_resolution), 1).astype(int)
            self.batch_nsteps.append(int(np.sum(row_nsteps)))

        self.batch_row_keys = []
        self.batch_event_global_ids = []
        self.batch_row_indices = []
        for batch_idxs in self.batch_traj_indices:
            rows_list = [self.trajectory_row_indices[t] for t in batch_idxs]
            if len(rows_list) == 0:
                self.batch_row_keys.append(np.empty((0,), dtype=self.traj_keys.dtype))
                self.batch_event_global_ids.append(np.empty((0,), dtype=np.int64))
                self.batch_row_indices.append(np.empty((0,), dtype=np.int64))
                continue

            rows = np.concatenate(rows_list)
            batch_keys = np.ascontiguousarray(np.unique(self.tracks_struct[rows][['eventID', 'trackID']]))
            batch_event_ids = np.asarray(np.unique(batch_keys['eventID']), dtype=np.int64)
            self.batch_row_keys.append(batch_keys)
            self.batch_event_global_ids.append(batch_event_ids)
            self.batch_row_indices.append(np.asarray(rows, dtype=np.int64))

        self.max_batch_nsteps = int(max(self.batch_nsteps)) if len(self.batch_nsteps) > 0 else 0

        self.chopped = chopped
        self.pad = pad
        self.electron_sampling_resolution = electron_sampling_resolution
        self.tot_data_length = tot_data_length if 'tot_data_length' in locals() else 0
        self.print_input = print_input
        self._evt_col = self.track_fields.index("eventID")
        zero_pad_fields = ["n_electrons", "dE", "dEdx", "dx", "long_diff", "tran_diff"]
        neg_one_pad_fields = ["trackID", "pixel_plane"]
        self._zero_pad_cols = [self.track_fields.index(name) for name in zero_pad_fields if name in self.track_fields]
        self._neg_one_pad_cols = [self.track_fields.index(name) for name in neg_one_pad_fields if name in self.track_fields]
        logger.info(f"-- The used simulation data includes a total track length of {self.tot_data_length} cm.")
        logger.info(f"-- The number of simulation batches is {len(self.batch_traj_indices)}.")

    def _invalidate_rows(self, batch_arr, row_mask):
        if row_mask is None or not np.any(row_mask):
            return batch_arr
        batch_arr[row_mask, self._evt_col] = -1
        if self._zero_pad_cols:
            batch_arr[np.ix_(row_mask, self._zero_pad_cols)] = 0
        if self._neg_one_pad_cols:
            batch_arr[np.ix_(row_mask, self._neg_one_pad_cols)] = -1
        return batch_arr

    def pad_batch(self, batch_arr, target_len, idx=None):
        """Pad/sanitize a batch at dataset-level so callers don't duplicate logic."""
        batch_arr = np.asarray(batch_arr, dtype=np.float32).copy()
        cur_len = batch_arr.shape[0]

        if target_len > cur_len:
            batch_arr = np.pad(batch_arr, ((0, target_len - cur_len), (0, 0)), mode='constant', constant_values=0)
            pad_mask = np.zeros((target_len,), dtype=bool)
            pad_mask[cur_len:] = True
            batch_arr = self._invalidate_rows(batch_arr, pad_mask)

        if idx is not None:
            valid_local_event_ids = np.arange(len(self.batch_event_global_ids[idx]), dtype=np.int64)
            invalid_local_mask = (batch_arr[:, self._evt_col] >= 0) & (~np.isin(batch_arr[:, self._evt_col].astype(np.int64), valid_local_event_ids))
            batch_arr = self._invalidate_rows(batch_arr, invalid_local_mask)

        return batch_arr

    def __len__(self):
        return len(self.batch_traj_indices)

    def __getitem__(self, idx):
        """Construct and return the batch at index idx on-demand (NumPy array)."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Batch index out of range")
        traj_indices = self.batch_traj_indices[idx]
        # collect row indices for all trajectories in this batch
        rows_list = [self.trajectory_row_indices[t] for t in traj_indices]
        if len(rows_list) == 0:
            return np.empty((0, len(self.track_fields)), dtype=np.float32)
        rows = np.concatenate(rows_list)
        # select structured rows and convert to 2D float array
        selected_struct = self.tracks_struct[rows]
        batch_arr = np_from_structured(selected_struct)
        batch_arr = remap_event_ids_to_local(batch_arr, selected_struct['eventID'], self.batch_event_global_ids[idx], self.track_fields)

        if self.chopped:
            batch_arr = chop_tracks(batch_arr, self.track_fields, precision=self.electron_sampling_resolution)

        # pad to global max
        if self.pad and self.max_batch_nsteps > 0:
            cur_len = batch_arr.shape[0]
            if cur_len < self.max_batch_nsteps:
                batch_arr = self.pad_batch(batch_arr, self.max_batch_nsteps, idx)

        if self.print_input:
            logger.info(f"Yielding simulation batch {idx} shape {batch_arr.shape}")
            logger.info(f"Simulation ['eventID', 'trackID'] batch: {self.get_batch_row_keys()[idx].tolist()}")

        return np.asarray(batch_arr, dtype=np.float32)

    def get_track_fields(self):
        return self.track_fields

    def get_batch_global_event_ids(self, idx=None):
        if idx is None:
            return self.batch_event_global_ids
        return self.batch_event_global_ids[idx]

    def get_batch_row_keys(self):
        """Return row-level (eventID, trackID) keys for all rows used in batch idx."""
        return self.batch_row_keys

    def get_batch_row_indices(self, idx=None):
        """Return source row indices in the original segments table for each batch."""
        if idx is None:
            return self.batch_row_indices
        return self.batch_row_indices[idx]

class TgtTracksDataset:
    def __init__(self, filename, dataset_sim, swap_xz=True, chopped=True, pad=True, electron_sampling_resolution=0.1, print_input=False):
        self.filename = filename
        self.chopped = chopped
        self.pad = pad
        self.electron_sampling_resolution = electron_sampling_resolution
        self.print_input = print_input
        self.swap_xz = swap_xz

        self.sim_track_fields = dataset_sim.get_track_fields()
        with h5py.File(self.filename, "r") as f:
            tracks_ds = f["segments"][:] # convert to array

        if not 't0' in tracks_ds.dtype.names:
            tracks_ds = rfn.append_fields(tracks_ds, 't0', np.zeros(tracks_ds.shape[0]), usemask=False)

        self.tgt_track_fields = tracks_ds.dtype.names

        replace_map = {
            'event_id': 'eventID',
            'traj_id': 'trackID',
        }
        self.tgt_track_fields = tuple([replace_map.get(field, field) for field in self.tgt_track_fields])

        tracks_ds.dtype.names = self.tgt_track_fields

        if self.swap_xz:
            x_start = np.copy(tracks_ds['x_start'])
            x_end = np.copy(tracks_ds['x_end'])
            x = np.copy(tracks_ds['x'])

            tracks_ds['x_start'] = np.copy(tracks_ds['z_start'])
            tracks_ds['x_end'] = np.copy(tracks_ds['z_end'])
            tracks_ds['x'] = np.copy(tracks_ds['z'])

            tracks_ds['z_start'] = x_start
            tracks_ds['z_end'] = x_end
            tracks_ds['z'] = x

        # Keep target rows in memory once and build a fast row index by key.
        self.tracks_struct = tracks_ds
        tgt_key_ids = self._pack_evt_trk_ids(self.tracks_struct['eventID'], self.tracks_struct['trackID'])
        self._target_sorted_row_idx = np.argsort(tgt_key_ids)
        sorted_key_ids = tgt_key_ids[self._target_sorted_row_idx]
        self._target_unique_key_ids, self._target_unique_starts = np.unique(sorted_key_ids, return_index=True)
        self._target_unique_ends = np.append(self._target_unique_starts[1:], len(self._target_sorted_row_idx))

        # Precompute a small per-batch key list from dataset_sim (int)
        self.batch_keys = dataset_sim.get_batch_row_keys()
        self.batch_event_global_ids = dataset_sim.get_batch_global_event_ids()

        self.batch_row_indices = []
        self.batch_nsteps = []
        self.tot_data_length = 0

        for keys in self.batch_keys:
            if keys.size == 0:
                self.batch_row_indices.append(np.empty((0,), dtype=int))
                self.batch_nsteps.append(0)
                continue

            key_ids = np.unique(self._pack_evt_trk_ids(keys['eventID'], keys['trackID']))
            row_idx = self._rows_for_key_ids(key_ids) 
            self.batch_row_indices.append(row_idx)

            if row_idx.size == 0:
                self.batch_nsteps.append(0)
            else:
                dx_vals = self.tracks_struct[row_idx]['dx']
                nsteps = np.maximum(np.ceil(dx_vals / self.electron_sampling_resolution), 1).astype(int)
                self.batch_nsteps.append(int(np.sum(nsteps)))
                self.tot_data_length += np.sum(dx_vals)

        self.max_batch_nsteps = int(max(self.batch_nsteps)) if self.batch_nsteps else 0

        # Basic metadata
        self.n_batches = len(self.batch_keys)
        self._evt_col = self.tgt_track_fields.index("eventID")
        zero_pad_fields = ["n_electrons", "dE", "dEdx", "dx", "long_diff", "tran_diff"]
        neg_one_pad_fields = ["trackID", "pixel_plane"]
        self._zero_pad_cols = [self.tgt_track_fields.index(name) for name in zero_pad_fields if name in self.tgt_track_fields]
        self._neg_one_pad_cols = [self.tgt_track_fields.index(name) for name in neg_one_pad_fields if name in self.tgt_track_fields]

        if self.print_input:
            logger.info(f"TgtTracksDataset prepared {self.n_batches} batch keys from sim dataset")

        logger.info(f"-- The used target data includes a total track length of {self.tot_data_length} cm.")
        logger.info(f"-- The number of target batches is {len(self.batch_keys)}.")

    def _invalidate_rows(self, batch_arr, row_mask):
        if row_mask is None or not np.any(row_mask):
            return batch_arr
        batch_arr[row_mask, self._evt_col] = -1
        if self._zero_pad_cols:
            batch_arr[np.ix_(row_mask, self._zero_pad_cols)] = 0
        if self._neg_one_pad_cols:
            batch_arr[np.ix_(row_mask, self._neg_one_pad_cols)] = -1
        return batch_arr

    def pad_batch(self, batch_arr, target_len, idx=None):
        """Pad/sanitize a target batch at dataset-level."""
        batch_arr = np.asarray(batch_arr, dtype=np.float32).copy()
        cur_len = batch_arr.shape[0]

        if target_len > cur_len:
            batch_arr = np.pad(batch_arr, ((0, target_len - cur_len), (0, 0)), mode='constant', constant_values=0)
            pad_mask = np.zeros((target_len,), dtype=bool)
            pad_mask[cur_len:] = True
            batch_arr = self._invalidate_rows(batch_arr, pad_mask)

        if idx is not None:
            valid_local_event_ids = np.arange(len(self.batch_event_global_ids[idx]), dtype=np.int64)
            invalid_local_mask = (batch_arr[:, self._evt_col] >= 0) & (~np.isin(batch_arr[:, self._evt_col].astype(np.int64), valid_local_event_ids))
            batch_arr = self._invalidate_rows(batch_arr, invalid_local_mask)

        return batch_arr

    @staticmethod
    def _pack_evt_trk_ids(event_ids, track_ids):
        """Pack (eventID, trackID) pairs into collision-free uint64 keys."""
        evt_u = np.asarray(event_ids, dtype=np.int64).astype(np.uint64)
        trk_u = np.asarray(track_ids, dtype=np.int64).astype(np.uint64)
        return (evt_u << np.uint64(32)) | (trk_u & np.uint64(0xFFFFFFFF))

    @staticmethod
    def _format_key_pairs_from_ids(key_ids, max_items=8):
        """Decode packed key ids into human-readable (eventID, trackID) pairs."""
        if key_ids.size == 0:
            return []
        key_ids = np.asarray(key_ids, dtype=np.uint64)
        evt = (key_ids >> np.uint64(32)).astype(np.int64)
        trk = (key_ids & np.uint64(0xFFFFFFFF)).astype(np.int64)
        n = min(max_items, key_ids.size)
        return list(zip(evt[:n].tolist(), trk[:n].tolist()))

    def _assert_batch_key_match(self, idx, expected_key_ids, selected_rows):
        """Ensure selected target rows contain exactly the expected batch keys."""
        got_key_ids = np.unique(self._pack_evt_trk_ids(selected_rows['eventID'], selected_rows['trackID']))

        if not np.array_equal(expected_key_ids, got_key_ids):
            missing = np.setdiff1d(expected_key_ids, got_key_ids, assume_unique=True)
            extra = np.setdiff1d(got_key_ids, expected_key_ids, assume_unique=True)
            msg = (
                f"Batch {idx} key mismatch between sim and tgt. "
                f"expected={expected_key_ids.size}, got={got_key_ids.size}, "
                f"missing={missing.size}, extra={extra.size}. "
                f"missing_sample={self._format_key_pairs_from_ids(missing)}, "
                f"extra_sample={self._format_key_pairs_from_ids(extra)}"
            )
            raise ValueError(msg)
    
    def _rows_for_key_ids(self, key_ids):
        left = np.searchsorted(self._target_unique_key_ids, key_ids, side='left')
        right = np.searchsorted(self._target_unique_key_ids, key_ids, side='right')

        missing_mask = left == right
        if missing_mask.any():
            missing_key_ids = key_ids[missing_mask]
            pairs = self._format_key_pairs_from_ids(missing_key_ids, max_items=16)
            raise ValueError(
                f"{missing_mask.sum()} of {len(key_ids)} sim key(s) are absent from the target file. "
                f"Missing (eventID, trackID) sample: {pairs}"
            )

        chunks = []
        for l, r in zip(left, right):
            for u in range(l, r):
                s = self._target_unique_starts[u]
                e = self._target_unique_ends[u]
                chunks.append(self._target_sorted_row_idx[s:e])

        if not chunks:
            return np.empty((0,), dtype=int)
        return np.sort(np.concatenate(chunks))

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_batches:
            raise IndexError("TgtTracksDataset index out of range")

        row_idx = self.batch_row_indices[idx]
        if row_idx.size == 0:
            return np.empty((0, len(self.tgt_track_fields)), dtype=np.float32)

        selected_rows = self.tracks_struct[row_idx]
        batch_arr = np_from_structured(selected_rows)
        batch_arr = remap_event_ids_to_local(batch_arr, selected_rows['eventID'], self.batch_event_global_ids[idx], self.tgt_track_fields)

        if self.chopped and batch_arr.size:
            batch_arr = chop_tracks(batch_arr, self.tgt_track_fields, precision=self.electron_sampling_resolution)

        if self.pad and self.max_batch_nsteps > 0 and batch_arr.shape[0] < self.max_batch_nsteps:
            batch_arr = self.pad_batch(batch_arr, self.max_batch_nsteps, idx)

        if self.print_input:
            logger.info(f"Yielding target batch {idx} shape {batch_arr.shape}")
            logger.info(f"selected_rows[['eventID', 'trackID']]: {self.batch_keys[idx].tolist()}")

        return np.asarray(batch_arr, dtype=np.float32)

    def get_track_fields(self):
        return self.tgt_track_fields

    def get_batch_global_event_ids(self, idx=None):
        if idx is None:
            return self.batch_event_global_ids
        return self.batch_event_global_ids[idx]
