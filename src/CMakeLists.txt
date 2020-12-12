add_executable(broxflow
        main.cpp
        opticalFlow.cpp)

target_link_libraries(broxflow
        ${OpenCV_LIBS})