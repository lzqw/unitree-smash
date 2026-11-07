"""Reserve physical disk blocks for final native evidence; never claim success on low space."""
import json
import os
from pathlib import Path
import shutil


class NativeOutputReserve:
    def __init__(self,output,*,reserve_bytes=512*1024**2,minimum_free_bytes=1536*1024**2,pressure_bytes=256*1024**2):
        self.output=Path(output);self.path=self.output/'.native_final_output.reserve'
        self.pressure_bytes=pressure_bytes;self.released=False
        self.output.mkdir(parents=True,exist_ok=True)
        free=shutil.disk_usage(self.output).free
        if free<minimum_free_bytes or minimum_free_bytes<reserve_bytes+pressure_bytes:
            raise OSError('Insufficient free space for native output plus final evidence reserve')
        with self.path.open('xb') as f:
            os.posix_fallocate(f.fileno(),0,reserve_bytes);f.flush();os.fsync(f.fileno())
        self.receipt=dict(admitted_free_bytes=free,reserve_bytes=reserve_bytes,
            minimum_free_bytes=minimum_free_bytes,pressure_bytes=pressure_bytes,
            reserve_blocks=self.path.stat().st_blocks,final_evidence_reserved=True)
        (self.output/'storage_admission.json').write_text(json.dumps(self.receipt,indent=2)+'\n')

    def check(self):
        if shutil.disk_usage(self.output).free<self.pressure_bytes:
            self.release()
            raise OSError('Storage pressure: released final evidence reserve and stopped native diagnostic')

    def release(self):
        if not self.released:
            self.path.unlink(missing_ok=True);self.released=True


def require_free_space(path,minimum=1536*1024**2):
    if shutil.disk_usage(path).free<minimum:raise OSError('Native job storage admission failed')
