glvs
fid = fopen('11.txt', 'r');
[dd, mn] = fread(fid, inf, 'uchar'); dd(3:3:end) = [];  % 读文件去空格
fclose(fid);
sz = fix(length(dd)/8);
dd1 = reshape(dd(1:sz*8),8,sz)'; dd1 = char(dd1);  % 转化为16进制字符形式
dd2 = typecast(uint32(hex2dec(dd1(:,[7,8,5,6,3,4,1,2]))),'single');  % 转化为单精度浮点
sz = fix(length(dd2)/34);
dd2 = reshape(dd2(1:34*sz),34,sz);  dd2 = dd2';  % 按34列分组
imu = [[dd2(:,3:5)*glv.dps,dd2(:,6:8)]*0.01,dd2(:,2)]; imuplot(imu,1);
return
