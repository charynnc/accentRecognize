"""Small script to test the dataloader module and inspect per-speaker split stats."""

import argparse

from dataloader import get_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Root data directory (e.g. data/)', default='/home1/chenhaoyang/data/accentDB/')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--report-limit', type=int, default=5, help='How many speakers to print in the summary')
    parser.add_argument('--skip-batch', action='store_true', help='Skip loading audio and only print metadata summary')
    args = parser.parse_args()

    dl = get_dataloader(args.data_dir,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        split=args.split)

    if not args.skip_batch:
        batch = next(iter(dl))
        print(f"Split: {args.split}")
        print('inputs.shape:', batch['inputs'].shape)
        print('input_lengths:', batch['input_lengths'])
        print('paths sample:', batch['paths'])
        print('speakers sample:', batch['speakers'])
        print('accents sample:', batch['accents'])
        print('accent_onehot:', batch['accent_onehot'])
        print('accent_indices:', batch['accent_indices'])
        print('mean per batch:', float(batch['inputs'].mean().item()))

    ds = dl.dataset

    print(f"Total samples in split '{args.split}':", len(ds))


if __name__ == '__main__':
    main()
