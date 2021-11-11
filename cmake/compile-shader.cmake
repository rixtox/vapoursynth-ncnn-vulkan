macro(compile_shader OUTPUT_LIST SHADERS_SRC_FILES)
    set(SHADERS_SRC_FILES ${SHADERS_SRC_FILES} ${ARGN})
    foreach(SHADER_SRC IN LISTS SHADERS_SRC_FILES)
        get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
        set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.hex.h)
        add_custom_command(
            OUTPUT ${SHADER_SPV_HEX_FILE}
            COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
            ARGS -V -s -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC}
            DEPENDS ${SHADER_SRC}
            COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
            VERBATIM
        )
        set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
        list(APPEND ${OUTPUT_LIST} ${SHADER_SPV_HEX_FILE})

        # fp16 storage
        set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")

        set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
        add_custom_command(
            OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
            COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
            ARGS -DNCNN_fp16_storage=1 -V -s -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC}
            DEPENDS ${SHADER_SRC}
            COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
            VERBATIM
        )
        set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
        list(APPEND ${OUTPUT_LIST} ${SHADER_fp16s_SPV_HEX_FILE})

        # int8 storage
        set(SHADER_int8s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_int8s")

        set(SHADER_int8s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_int8s_SRC_NAME_WE}.spv.hex.h)
        add_custom_command(
            OUTPUT ${SHADER_int8s_SPV_HEX_FILE}
            COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
            ARGS -DNCNN_fp16_storage=1 -DNCNN_int8_storage=1 -V -s -x -o ${SHADER_int8s_SPV_HEX_FILE} ${SHADER_SRC}
            DEPENDS ${SHADER_SRC}
            COMMENT "Building SPIR-V module ${SHADER_int8s_SRC_NAME_WE}.spv"
            VERBATIM
        )
        set_source_files_properties(${SHADER_int8s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
        list(APPEND ${OUTPUT_LIST} ${SHADER_int8s_SPV_HEX_FILE})
    endforeach()
endmacro()
