import gcsfs

fs = gcsfs.GCSFileSystem(project='hedge-fund-494103')

# List all folders under raw/orderbook/ to see what's there
folders = fs.ls('hedge-fund-494103-marketdata-mumbai/raw/orderbook/')
print(f"Total symbols in GCS: {len(folders)}")
print("\nSample folders:")
for f in sorted(folders)[:20]:
    print(f"  {f.split('/')[-1]}")

print("\n--- Searching for options ---")
all_folders = [f.split('/')[-1] for f in folders]
options = [f for f in all_folders if f.endswith(('CE', 'PE'))]
futures = [f for f in all_folders if f.endswith('FUT')]
print(f"Futures folders: {len(futures)}")
print(f"Options folders: {len(options)}")
if options:
    print("\nOptions symbols found:")
    for o in sorted(options)[:10]:
        print(f"  {o}")
