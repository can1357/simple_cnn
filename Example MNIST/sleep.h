#include <time.h>

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>

void nanosleep(const struct timespec *req, struct timespec *rem)
{
	Sleep((DWORD) (req.tv_sec * 1000)
	      + (DWORD) (req.tv_nsec != 0
			 ? req.tv_nsec / 1000000
			 : 0));
}
#endif
