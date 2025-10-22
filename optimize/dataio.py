import h5py
import numpy as np
from numpy.lib import recfunctions as rfn
import random
import logging
import jax.numpy as jnp
from jax import vmap
import jax
from typing import List, Union, Tuple
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.index = 0
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            self.indices = random.permutation(random.PRNGKey(self.seed), self.indices)

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            self.indices = random.permutation(random.PRNGKey(self.seed), self.indices)
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

        # Convert the batch to a JAX array if needed
        batch = jnp.array(batch)

        self.index += self.batch_size
        return batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def jax_from_structured(tracks):
    tracks_np = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)
    return jnp.array(tracks_np).astype(jnp.float32)

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
    length = np.sqrt(np.sum(segment**2, axis=1, keepdims=True))
    eps = 1e-10
    direction = segment / (length + eps)
    nsteps = np.maximum(np.ceil(length / precision), 1).astype(int).flatten()
    # step_size = length/nsteps
    new_tracks = np.vstack([split_track(tracks[i], nsteps[i], length[i], direction[i], i) for i in range(tracks.shape[0])])
    return new_tracks

def pad_sequence(sequences: List[jnp.ndarray],
                 padding_value: Union[float, int] = 0.0) -> jnp.ndarray:
    logger.info(f"Padding sequences with padding value: {padding_value}")
    # Input Validation
    if not isinstance(sequences, (list, tuple)):
        raise TypeError(f"Input 'sequences' must be a list or tuple of jnp arrays, got {type(sequences)}")

    if not sequences:
        return jnp.array([], dtype=jnp.float32)

    shapes = [s.shape for s in sequences]

    max_len = max(s[0] for s in shapes)

    def pad_single_sequence(seq, max_len):
        current_len = seq.shape[0]
        pad_len = max_len - current_len
        padded_seq = jnp.pad(seq, ((0, pad_len), (0, 0)), constant_values=padding_value)
        return padded_seq

    padded_sequences = [pad_single_sequence(seq, max_len) for seq in sequences]

    return padded_sequences


class TracksDataset:
    def __init__(self, filename, ntrack, max_nbatch=None, swap_xz=True, seed=3, random_ntrack=False, track_len_sel=2., 
                 max_abs_costheta_sel=0.966, min_abs_segz_sel=15., track_z_bound=28., max_batch_len=None, print_input=False,
                 chopped=True, pad=True, electron_sampling_resolution=0.001, live_selection=False):

        with h5py.File(filename, 'r') as f:
            tracks = np.array(f['segments'])

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
        # assuming tracks are in orders for trajectories
        length_load_threshold = max_batch_len * (max_nbatch + 2)
        loaded_tracks = tracks[np.cumsum(tracks['dx']) < length_load_threshold]
        tracks = loaded_tracks
        
        if live_selection:
            # flat index for all reasonable track [eventID, trackID] 
            index = []
            all_tracks = []

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

            keys = np.ascontiguousarray(selected_tracks[['eventID', 'trackID']])
            all_tracks = jax_from_structured(selected_tracks[mask])
            index, inverse_idx = np.unique(keys, return_inverse=True)
        else:
            keys = np.ascontiguousarray(tracks[['eventID', 'trackID']])
            all_tracks = jax_from_structured(tracks)
            index, inverse_idx = np.unique(keys, return_inverse=True)

        # all fit with a sub-set of tracks
        dict_fit_tracks = defaultdict(list)
        random.seed(seed)
        fit_tracks = []
        if ntrack is None or ntrack >= len(index) or ntrack <= 0:
            fit_index = index
            if 'file_traj_id' in self.track_fields:
                sorted_tracks = all_tracks[all_tracks[:, self.track_fields.index('file_traj_id')].argsort()]
                unique_file_traj_id, counts = np.unique(sorted_tracks[:, self.track_fields.index('file_traj_id')], return_counts=True)
                traj_split = np.cumsum(counts)
                traj_split = np.append(traj_split, len(sorted_tracks))
                traj_split = np.insert(traj_split, 0, 0)
                for i in range(len(traj_split)-1):
                    fit_tracks.append(sorted_tracks[traj_split[i]:traj_split[i+1]])
            else:
                for traj_idx in range(len(index)):
                    mask = inverse_idx == traj_idx
                    fit_tracks.append(all_tracks[mask])
        else:
            if random_ntrack:
                index_idx_pool = np.random.randint(0, len(index), ntrack)
            else:
                index_idx_pool = np.arange(ntrack)

            fit_index = index[index_idx_pool]
            fit_tracks = []
            for traj_idx in index_idx_pool:
                mask = inverse_idx == traj_idx
                fit_tracks.append(all_tracks[mask])

        # Exclude [eventID, trackID] == [0, 0] even it's physical due to the pad value
        idx_0 = np.where((fit_index['eventID'] == 0) & (fit_index['trackID'] == 2))
        if len(idx_0[0]) > 0:
            del fit_tracks[idx_0[0][0]]
            fit_index = np.delete(fit_index, idx_0[0][0])

        if max_batch_len is not None:
            # Get track length per trajectory
            traj_len = np.array([traj[:, self.track_fields.index("dx")].sum() for traj in fit_tracks])

            remove_traj_index = np.where(traj_len > max_batch_len)

            lengths_ft = traj_len.copy()[traj_len <= max_batch_len]
            fit_tracks_ft = fit_tracks.copy()
            if len(remove_traj_index[0]) > 0:
                for i_remove in remove_traj_index[0]:
                    del fit_tracks_ft[i_remove]
                fit_index = np.delete(fit_index, remove_traj_index[0])
            
            # Cumulative sum to track segment lengths
            cumsum_lengths = np.cumsum(lengths_ft)

            # Find batch boundaries
            split_points = np.where(np.diff(np.floor_divide(cumsum_lengths, max_batch_len)) > 0)[0] + 1
            split_points = np.append(split_points, len(cumsum_lengths))
            split_points = np.insert(split_points, 0, 0)

            # Cap the number of batches if max_nbatch is set
            if max_nbatch:
                split_points = split_points[:max_nbatch]

            # Create JaX batches
            batches = []
            fit_index_bt = []
            for i in range(len(split_points)-1):
                batches.append(np.vstack(fit_tracks_ft[split_points[i]:split_points[i+1]]))
                fit_index_bt.append(fit_index[split_points[i]:split_points[i+1]])
            tot_data_length = cumsum_lengths[split_points[-1]-1]
            if chopped:
                fit_tracks = [jnp.array(chop_tracks(batch, self.track_fields, electron_sampling_resolution)) for batch in batches]
            else:
                fit_tracks = [jnp.array(batch) for batch in batches]

            if print_input:
                logger.info(f"training set [ev, trk]: {[idx.tolist() for idx in fit_index_bt]}")
            logger.info(f"-- The used simulation data includes a total track length of {tot_data_length} cm.")
            logger.info(f"-- The maximum batch track length is {max_batch_len} cm.")
            logger.info(f"-- The number of batches is {len(batches)}.")

        if pad:
            # Fix the pad value to 0, in case unexpected simulation behaviour
            self.tracks = pad_sequence(fit_tracks, padding_value = 0)
        else:
            self.tracks = fit_tracks

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx]
        
    def get_track_fields(self):
        return self.track_fields

class TgtTracksDataset:
    def __init__(self, filename, dataloader_sim, swap_xz=True, chopped=True, pad=True, electron_sampling_resolution=0.001):

        with h5py.File(filename, 'r') as f:
            tracks = np.array(f['segments'])

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


        self.tgt_track_fields = tracks.dtype.names
        replace_map = {
            'event_id': 'eventID',
            'traj_id': 'trackID',
        }
        self.tgt_track_fields = tuple([replace_map.get(field, field) for field in self.tgt_track_fields])

        tracks.dtype.names = self.tgt_track_fields

        self.sim_track_fields = dataloader_sim.get_track_fields()

        # Only load useful tracks
        batches = []
        tot_data_length = 0
        if 'file_traj_id' in self.sim_track_fields and 'file_traj_id' in self.tgt_track_fields:
            for bt in dataloader_sim:
                load_file_traj = np.unique(bt[:, self.sim_track_fields.index("file_traj_id")])
                index0 = np.where(load_file_traj==0)
                load_file_traj = np.delete(load_file_traj, index0)
                mask = np.isin(tracks['file_traj_id'], load_file_traj)

                # Explicitly check that the input tracks are the same for the target and the simulation
                if not np.allclose(tracks['file_traj_id'][mask], load_file_traj):
                    raise ValueError("Target and input do not contain the same tracks! Please check.")

                batches.append(np.vstack(jax_from_structured(tracks[mask])))
                tot_data_length += np.sum(tracks[mask]['dx'])
        else:
            for bt in dataloader_sim:
                if np.max(bt[:, self.sim_track_fields.index("trackID")]) > 1E5:
                    raise ValueError("Assumption broke! More than 1E5 trajectories in some event!")
                load_file_traj = np.unique(bt[:, self.sim_track_fields.index("eventID")]*1E5 + bt[:, self.sim_track_fields.index("trackID")])
                event_track_id = tracks['eventID']*1E5 + tracks['trackID']
                mask = np.isin(event_track_id, load_file_traj)
 
                # Explicitly check that the input tracks are the same for the target and the simulation
                if not np.allclose(event_track_id[mask], load_file_traj):
                    raise ValueError("Target and input do not contain the same tracks! Please check.")

                batches.append(np.vstack(jax_from_structured(tracks[mask])))
                tot_data_length += np.sum(tracks[mask]['dx'])
 
        if chopped:
            fit_tracks = [jnp.array(chop_tracks(batch, self.tgt_track_fields, electron_sampling_resolution)) for batch in batches]
        else:
            fit_tracks = [jnp.array(batch) for batch in batches]

        logger.info(f"-- The used target data includes a total track length of {tot_data_length} cm.")
        logger.info(f"-- The number of target batches is {len(batches)}.")

        if pad:
            self.tracks = pad_sequence(fit_tracks, padding_value = 0)
        else:
            self.tracks = fit_tracks

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx]

    def get_track_fields(self):
        return self.tgt_track_fields
