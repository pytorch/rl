@echo on
set CL=/I"C:\Program Files (x86)\torchrl\include"
msbuild "-p:Configuration=Release" "-p:BuildInParallel=true" "-p:MultiProcessorCompilation=true" "-p:CL_MPCount=%1" hello-world.vcxproj -maxcpucount:%1
