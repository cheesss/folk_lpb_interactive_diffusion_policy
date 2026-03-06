"""
Create a 2-episode HDF5 sub-dataset from a full robomimic-style HDF5.
Used to run fast sanity checks (e.g. 1-epoch training) without loading the full dataset.

Usage:
  python scripts/create_2ep_subset.py
  python scripts/create_2ep_subset.py --source /path/to/full.hdf5 --out /path/to/out_2ep.hdf5
"""
import argparse
import h5py


def copy_entity(src_ent, dst_parent, name):
    """Copy an h5py Dataset or Group from src_ent into dst_parent under name."""
    if isinstance(src_ent, h5py.Dataset):
        data = src_ent[()]
        dst_parent.create_dataset(name, data=data, dtype=data.dtype)
    else:
        grp = dst_parent.create_group(name)
        for k in src_ent.keys():
            copy_entity(src_ent[k], grp, k)


def main():
    parser = argparse.ArgumentParser(description="Create 2-episode subset HDF5")
    parser.add_argument(
        "--source",
        default="/media/son/son_dataset_2/IL/dataset/rb10_pick_and_place/pick_tissue_all.hdf5",
        help="Source HDF5 path",
    )
    parser.add_argument(
        "--out",
        default="/media/son/son_dataset_2/IL/dataset/rb10_pick_and_place/pick_tissue_2ep.hdf5",
        help="Output HDF5 path (2 episodes)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=2,
        help="Number of episodes to copy (default 2)",
    )
    args = parser.parse_args()

    with h5py.File(args.source, "r") as src:
        demos = src["data"]
        n_demos = len(demos)
        if n_demos < args.num_episodes:
            raise RuntimeError(
                f"Source has only {n_demos} demos; requested {args.num_episodes}"
            )
        demo_keys = [f"demo_{i}" for i in range(args.num_episodes)]

        with h5py.File(args.out, "w") as dst:
            data_dst = dst.create_group("data")
            for key in demo_keys:
                copy_entity(demos[key], data_dst, key)
                print(f"Copied {key}")

    print(f"Written {args.num_episodes} episodes to {args.out}")


if __name__ == "__main__":
    main()
