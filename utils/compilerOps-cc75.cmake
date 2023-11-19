message("-- Selecting compiler Ops...")

# Set library types to be shared
set(LIBRARY_TYPE SHARED)

# Set a default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set C/C++ standard
set(CMAKE_CXX_STANDARD 11)

# Define common C compiler flags
set(CMAKE_C_FLAGS "-DNOPRED")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "")

# Define common CXX compiler flags
set(CMAKE_CXX_FLAGS "-DNOPRED")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "")

# Define common Fortran compiler flags
set(CMAKE_Fortran_FLAGS "-DNOPRED")
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0")
set(CMAKE_Fortran_FLAGS_RELEASE "")

# Define specific compiler flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
	message("-- GNU compiler detected")
	if (USE_GPU)
		message(FATAL_ERROR "GPU not supported with GNU compiler")
	endif()
	# Common GNNU+MPI flags
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-cpp -DNOACC")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-cpp -DNOACC")
	set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-cpp -DNOACC -ffree-line-length-none")
	# Debug
	set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic")
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic")
	set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic -fbacktrace -Wconversion-extra -ftrapv -fcheck=all -ffpe-trap=invalid,zero,overflow")
	#set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "-Wall -Wextra -Wpedantic -fbacktrace -Wconversion-extra -ftrapv -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow")
	# Release
	set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-O3 -march=native")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-O3 -march=native")
	set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-O3 -march=native")
elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel" OR CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
	message("-- Intel compiler detected")
	if (USE_GPU)
		message(FATAL_ERROR "GPU not supported with Intel compiler")
	endif()
	# Common Intel+MPI flags
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-DNOACC")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-DNOACC")
	set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-fpp -DNOACC")
	# Debug
	set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} "-debug all")
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-debug all")
	set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "-traceback -check all -ftrapuv -debug all -fpe3 -fpe-all=3")
	# Release
	set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-O3")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-O3")
	set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-O3")
	if(USE_MN)
		message("Optimizing for MN4")
		set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-xCORE-AVX512 -mtune=skylake")
		set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-xCORE-AVX512 -mtune=skylake")
		set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-xCORE-AVX512 -mtune=skylake")
	else()
		set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-mtune=native")
		set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-mtune=native")
		set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-mtune=native")
	endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "NVHPC" OR CMAKE_C_COMPILER_ID STREQUAL "PGI")
	message("-- NVHPC compiler detected")
	# Common NVHPC+MPI flags
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-cpp -lstdc++ -D_USE_NVTX -lnvToolsExt -cuda -Minfo=accel")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-cpp -lstdc++ -D_USE_NVTX -lnvToolsExt -cuda -Minfo=accel")
	set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-cpp -lstdc++ -D_USE_NVTX -lnvToolsExt -cuda -Minfo=accel")
	# Debug
	set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} "-Minform=inform -C -Mbounds -Mchkptr -traceback -Ktrap=fp,unf")
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-Minform=inform -C -Mbounds -Mchkstk -traceback -Ktrap=fp,unf")
	set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "-Minform=inform -C -Mbounds -Mchkstk -traceback -Ktrap=fp,unf")
	# Release
	set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-fast")
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-fast")
	set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-fast")
	# GPU options
	if(USE_GPU AND USE_MEM_MANAGED)
		message("Setting up ACC flags with managed memory")
		if(USE_PCPOWER)
			message("Setting up ACC flags with CC70")
			set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-gpu=cc70,cuda10.2,managed,lineinfo -acc")
			set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-gpu=cc70,cuda10.2,managed,lineinfo -acc")
			set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-gpu=cc70,cuda10.2,managed,lineinfo -acc")
		else()
			message("IMPORTANT: Verify the device compute capability!!!")
			set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-gpu=cc75,managed,lineinfo -cuda -acc")
			set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-gpu=cc75,managed,lineinfo -cuda -acc")
			set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-gpu=cc75,managed,lineinfo -cuda -acc")
		endif()
	elseif(USE_GPU)
		message("Setting up ACC flags")
		if(USE_PCPOWER)
			message("Setting up ACC flags with CC70")
			set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-gpu=cc70,cuda10.2,lineinfo -cuda -acc")
			set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-gpu=cc70,cuda10.2,lineinfo -cuda -acc")
			set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-gpu=cc70,cuda10.2,lineinfo -cuda -acc")
		else()
			message("IMPORTANT: Verify the device compute capability!!!")
			set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-gpu=cc75,lineinfo -cuda -acc")
			set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-gpu=cc75,lineinfo -cuda -acc")
			set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-gpu=cc75,lineinfo -cuda -acc")
		endif()
	else()
		#NVIDIA COMPILERS WITHOUT GPUs
		set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-DNOACC")
		set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-DNOACC")
		set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-DNOACC")
		# Debug
		set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} "-DNOACC")
		set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-DNOACC")
		set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "-DNOACC")
		# Release
		set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "-DNOACC")
		set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-DNOACC")
		set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "-DNOACC")
	endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        message("-- Cray compiler detected")
        # Common Cray+MPI flags
        set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "")
        set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "")
        set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-eZ")
        # Debug
        set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} "")
        set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "")
        set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} "")
        # Release
        set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} "")
        set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "")
        set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} "")
        # GPU options
        if(USE_GPU)
                message("Setting up ACC flags")
                message("IMPORTANT: Verify the device compute capability!!!")
                set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "")
                set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "")
                set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} "-hacc -hlist=a -target-accel=amd_gfx90a")
        else()
                message("AMD without GPUs is still not available")
                #NVIDIA COMPILERS WITHOUT GPUs
                set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS} "-DNOACC")
                set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS} "-DNOACC")
                set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS} "-DNOACC")
        endif()
else()
	message("this shit: " ${CMAKE_C_COMPILER_ID})
	message(FATAL_ERROR "Unknown compiler")
endif()

# Adjust stringg so ; is removed from the command
string(REPLACE ";" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE ";" " " CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}")
string(REPLACE ";" " " CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
string(REPLACE ";" " " CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG}")
string(REPLACE ";" " " CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE ";" " " CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE}")
