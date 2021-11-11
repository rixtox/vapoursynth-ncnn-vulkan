#define FreeAndClear(p) \
    if ((p) != nullptr && (*(p)) != nullptr) { delete (*(p)); (*(p)) = nullptr; }

int tryCreateGpuInstance();
void tryDestoryGpuInstance();
