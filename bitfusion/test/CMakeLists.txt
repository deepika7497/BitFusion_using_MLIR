configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(BITFUSION_TEST_DEPENDS
        FileCheck count not
        bitfusion-capi-test
        bitfusion-opt
        bitfusion-translate
        )

add_lit_testsuite(check-bitfusion "Running the bitfusion regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${BITFUSION_TEST_DEPENDS}
        )
set_target_properties(check-bitfusion PROPERTIES FOLDER "Tests")

add_lit_testsuites(BITFUSION ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BITFUSION_TEST_DEPENDS})

add_subdirectory(CAPI)
