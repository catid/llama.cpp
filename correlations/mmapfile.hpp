#include <cstdint.h>


//------------------------------------------------------------------------------
// Memory-mapped file

struct MappedView;

/// This represents a file on disk that will be mapped
struct MappedFile
{
    friend struct MappedView;

#if defined(_WIN32)
    /*HANDLE*/ void* File = nullptr;
#else
    int File = -1;
#endif

    bool ReadOnly = true;
    uint64_t Length = 0;

    inline bool IsValid() const { return Length != 0; }

    // Opens the file for shared read-only access with other applications
    // Returns false on error (file not found, etc)
    bool OpenRead(
        const char* path,
        bool read_ahead = false,
        bool no_cache = false);

    // Creates and opens the file for exclusive read/write access
    bool OpenWrite(
        const char* path,
        uint64_t size);

    // Resizes a file
    bool Resize(uint64_t size);

    void Close();

    MappedFile();
    ~MappedFile();
};


//------------------------------------------------------------------------------
// MappedView

/// View of a portion of the memory mapped file
struct MappedView
{
    void* Map = nullptr;
    MappedFile* File = nullptr;
    uint8_t* Data = nullptr;
    uint64_t Offset = 0;
    uint32_t Length = 0;

    // Returns false on error
    bool Open(MappedFile* file);

    // Returns 0 on error, 0 length means whole file
    uint8_t* MapView(uint64_t offset = 0, uint32_t length = 0);

    void Close();

    MappedView();
    ~MappedView();
};
