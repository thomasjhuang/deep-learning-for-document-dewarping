import os, argparse

def main():
    parser = argparse.ArgumentParser('Remove all files of according extension')
    parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='./A')
    parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='./B')
    parser.add_argument('--ext', dest='ext', help='file extension you want removed', type=str, default='.png')
    args = parser.parse_args()
    for root, dirs, files in os.walk(args.fold_A):
        for file in files:
            if file.endswith(args.ext):
                os.remove(os.path.join(root,file))
    for root, dirs, files in os.walk(args.fold_B):
        for file in files:
            if file.endswith(args.ext):
                os.remove(os.path.join(root,file))
    print(args.ext + "removed")

if __name__ == "__main__":
    main()
