cmake_minimum_required(VERSION 3.21)

project(
    neural_cpp
    VERSION 0.1.0
    DESCRIPTION "A neural network framework in modern C++"
    LANGUAGES CXX
)

include(CTest)

option(NEURAL_CPP_BUILD_TESTS "Build unit tests" ON)
option(NEURAL_CPP_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(NEURAL_CPP_ENABLE_INSTALL "Enable install rules" OFF)

add_library(neural_cpp)

target_sources(neural_cpp
    PRIVATE
        src/model.cpp
        src/linear.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/layer.h
            include/linear.h
            include/model.h
            include/tensor.h
)

target_include_directories(neural_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(neural_cpp PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(neural_cpp
        PRIVATE
            /W4
            /permissive-
            $<$<BOOL:${NEURAL_CPP_WARNINGS_AS_ERRORS}>:/WX>
    )
else()
    target_compile_options(neural_cpp
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wold-style-cast
            -Wnon-virtual-dtor
            -Woverloaded-virtual
            -Wnull-dereference
            -Wdouble-promotion
            $<$<BOOL:${NEURAL_CPP_WARNINGS_AS_ERRORS}>:-Werror>
    )
endif()

target_compile_definitions(neural_cpp
    PUBLIC
        NEURAL_CPP_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        NEURAL_CPP_VERSION_MINOR=${PROJECT_VERSION_MINOR}
)

set_target_properties(neural_cpp PROPERTIES
    OUTPUT_NAME neural_cpp
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(NEURAL_CPP_BUILD_TESTS)
    enable_testing()

    add_executable(neural_cpp_tests
        tests/test_main.cpp
    )

    target_link_libraries(neural_cpp_tests
        PRIVATE
            neural_cpp
    )

    target_compile_features(neural_cpp_tests PRIVATE cxx_std_20)

    if(MSVC)
        target_compile_options(neural_cpp_tests
            PRIVATE
                /W4
                /permissive-
                $<$<BOOL:${NEURAL_CPP_WARNINGS_AS_ERRORS}>:/WX>
        )
    else()
        target_compile_options(neural_cpp_tests
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                $<$<BOOL:${NEURAL_CPP_WARNINGS_AS_ERRORS}>:-Werror>
        )
    endif()

    add_test(NAME neural_cpp_tests COMMAND neural_cpp_tests)
endif()

if(NEURAL_CPP_ENABLE_INSTALL)
    include(GNUInstallDirs)

    install(
        TARGETS neural_cpp
        EXPORT neural_cppTargets
        FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        EXPORT neural_cppTargets
        FILE neural_cppTargets.cmake
        NAMESPACE neural_cpp::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neural_cpp
    )
endif()
