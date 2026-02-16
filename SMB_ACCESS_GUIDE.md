# SMB Access Guide (cimec-storage/tmp_uri)

smb://cimec-storage/tmp_uri

## 1) Mount the SMB share

```bash
gio mount smb://cimec-storage/tmp_uri
gio mount -l
```

## 2) Go to the mounted path (GVFS)

```bash
cd /run/user/$UID/gvfs/smb-share:server=cimec-storage,share=tmp_uri
ls -la
```

## 3) Open the Outputs folder

```bash

cd /run/user/$UID/gvfs/smb-share:server=cimec-storage,share=tmp_uri/hd6tb/icaro/Outputs
ls -la
```

## 5) Copy files to this folder


```bash
cp /path/to/local/file.ext /run/user/$UID/gvfs/smb-share:server=cimec-storage,share=tmp_uri/hd6tb/icaro/Outputs/
cp -r /path/to/local/folder /run/user/$UID/gvfs/smb-share:server=cimec-storage,share=tmp_uri/hd6tb/icaro/Outputs/
```
## 6) Unmount the SMB share when done

```bash
gio mount -u smb://cimec-storage/tmp_uri
```

## Notes

- If you cannot see the path under /run/user/$UID/gvfs, remount with gio mount and recheck.
- If connection fails, verify network/VPN access to cimec-storage and retry.
