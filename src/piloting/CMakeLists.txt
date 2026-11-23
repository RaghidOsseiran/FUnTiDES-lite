add_library(piloting INTERFACE)

target_include_directories(piloting
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CMAKE_BINARY_DIR}/include
)

if(COMPILE_PILOTING)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  add_executable(pilote
    src/main.cc
  )

  target_include_directories(pilote
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_SOURCE_DIR}/external/cxxopts
  )

  target_link_libraries(pilote
    PRIVATE
      ${STDCXXFS_LIB}
  )

install(TARGETS pilote
        RUNTIME DESTINATION bin)

endif()
