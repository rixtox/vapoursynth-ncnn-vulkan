cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if (NOT DEFINED ncnn_ROOT)
    set(ncnn_ROOT ${CMAKE_CURRENT_LIST_DIR}/../vendor/ncnn)
    if (NOT EXISTS ${ncnn_ROOT})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../vendor)
        if (MSVC)
            set(_NCNN_BUILD_VERSION 20210720)
            set(_NCNN_BASE_URL https://github.com/Tencent/ncnn/releases/download/${_NCNN_BUILD_VERSION}/)
            if (MSVC_TOOLSET_VERSION GREATER_EQUAL 142)
                set(_NCNN_FILE_NAME ncnn-${_NCNN_BUILD_VERSION}-windows-vs2019)
            elseif (MSVC_TOOLSET_VERSION EQUAL 141)
                set(_NCNN_FILE_NAME ncnn-${_NCNN_BUILD_VERSION}-windows-vs2017)
            elseif (MSVC_TOOLSET_VERSION LESS_EQUAL 140)
                set(_NCNN_FILE_NAME ncnn-${_NCNN_BUILD_VERSION}-windows-vs2015)
            endif ()

            if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/../vendor/${_NCNN_FILE_NAME}.zip)
                file(DOWNLOAD
                    ${_NCNN_BASE_URL}${_NCNN_FILE_NAME}.zip
                    ${CMAKE_CURRENT_LIST_DIR}/../vendor/${_NCNN_FILE_NAME}.zip
                    SHOW_PROGRESS )
            endif ()

            if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/../vendor/${_NCNN_FILE_NAME})
                file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_LIST_DIR}/../vendor/${_NCNN_FILE_NAME}.zip
                    DESTINATION ${CMAKE_CURRENT_LIST_DIR}/../vendor/
                    VERBOSE )
            endif ()
            file(RENAME ${CMAKE_CURRENT_LIST_DIR}/../vendor/${_NCNN_FILE_NAME} ${ncnn_ROOT})
        endif ()
    endif ()

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ncnn_ROOT ${ncnn_ROOT}/x64/lib/cmake/ncnn)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(ncnn_ROOT ${ncnn_ROOT}/x86/lib/cmake/ncnn)
    endif()
endif ()
