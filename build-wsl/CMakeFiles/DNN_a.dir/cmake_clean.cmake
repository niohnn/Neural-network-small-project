file(REMOVE_RECURSE
  "libDNN.a"
  "libDNN.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/DNN_a.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
