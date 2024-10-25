1.将matrix_test文件夹放到测试设备路径下
2.进入matrix_test目录，执行chmod a+x klu
3. 执行命令 ： LD_LIBRARY_PATH=./lib_test ./KLU -c 9 -m 300 -r 100000
备注： -c是指定的cpu核  -m是指定求解的矩阵阶数  -r是需要重复求解的次数
4. 等待执行结果
5.m后参数改为135，执行135维矩阵的解算：LD_LIBRARY_PATH=./lib_test ./klu -c 9 -m 135 -r 100000


