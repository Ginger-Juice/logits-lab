import ctypes
cu=ctypes.WinDLL("nvcuda.dll")
v=ctypes.c_int(); print("cuInit",cu.cuInit(0)); cu.cuDriverGetVersion(ctypes.byref(v)); print("driver api version",v.value)
d=ctypes.c_int(); print("cuDeviceGet",cu.cuDeviceGet(ctypes.byref(d),0))
ctx=ctypes.c_void_p(); r=cu.cuCtxCreate_v2(ctypes.byref(ctx),0,d); print("cuCtxCreate",r)
if r==0:
    p=ctypes.c_uint64(); print("cuMemAlloc",cu.cuMemAlloc_v2(ctypes.byref(p),1<<20))