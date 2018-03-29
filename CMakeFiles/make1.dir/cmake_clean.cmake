file(REMOVE_RECURSE
  "make1.pdb"
  "make1"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/make1.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
