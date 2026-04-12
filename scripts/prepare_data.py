import zipfile
import os
from tqdm import tqdm

ZIP_PATH = './data/food-101.zip'
DST_DIR  = './data/food10'

CLASSES = [
    'pizza', 'sushi', 'hamburger', 'hot_dog', 'french_fries',
    'ice_cream', 'omelette', 'pancakes', 'ramen', 'steak'
]

# Create destination folders
for split in ['train', 'test']:
    for cls in CLASSES:
        os.makedirs(f'{DST_DIR}/{split}/{cls}', exist_ok=True)

print("Opening zip file...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    all_files = zf.namelist()

    # Step 1: detect the actual prefix inside the zip
    # We'll find where train.txt lives
    meta_path = None
    for f in all_files:
        if f.endswith('meta/train.txt'):
            meta_path = f
            break

    if not meta_path:
        print("ERROR: Could not find meta/train.txt inside the zip!")
        print("First 10 files in zip:")
        for f in all_files[:10]:
            print(f"  {f}")
        exit()

    # Detect prefix — everything before 'meta/train.txt'
    prefix = meta_path.replace('meta/train.txt', '')
    print(f"Detected zip prefix: '{prefix}'")

    # Step 2: read train.txt and test.txt directly from zip
    with zf.open(f'{prefix}meta/train.txt') as f:
        train_files = [l.decode().strip() for l in f.readlines()]
    with zf.open(f'{prefix}meta/test.txt') as f:
        test_files = [l.decode().strip() for l in f.readlines()]

    # Filter to our 10 classes
    train_10 = [f for f in train_files if f.split('/')[0] in CLASSES]
    test_10  = [f for f in test_files  if f.split('/')[0] in CLASSES]

    print(f"Files to extract: {len(train_10)} train | {len(test_10)} test")

    # Step 3: extract train images
    print("\nExtracting train images...")
    missing = 0
    for entry in tqdm(train_10):
        cls, fname = entry.split('/')
        zip_path = f'{prefix}images/{cls}/{fname}.jpg'
        try:
            data = zf.read(zip_path)
            with open(f'{DST_DIR}/train/{cls}/{fname}.jpg', 'wb') as out:
                out.write(data)
        except KeyError:
            missing += 1

    print(f"  Missing files: {missing}")

    # Step 4: extract test images
    print("\nExtracting test images...")
    missing = 0
    for entry in tqdm(test_10):
        cls, fname = entry.split('/')
        zip_path = f'{prefix}images/{cls}/{fname}.jpg'
        try:
            data = zf.read(zip_path)
            with open(f'{DST_DIR}/test/{cls}/{fname}.jpg', 'wb') as out:
                out.write(data)
        except KeyError:
            missing += 1

    print(f"  Missing files: {missing}")

# Summary
print("\n--- Dataset Summary ---")
for split in ['train', 'test']:
    total = 0
    for cls in CLASSES:
        count = len(os.listdir(f'{DST_DIR}/{split}/{cls}'))
        print(f"  {split}/{cls}: {count} images")
        total += count
    print(f"  Total {split}: {total}\n")

print("Done!")