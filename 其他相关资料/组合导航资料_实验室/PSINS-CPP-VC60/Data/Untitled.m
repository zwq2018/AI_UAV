glvs
psinstypedef(156);
fid=fopen('ins.bin'); avp=fread(fid, [16,inf], 'double')'; fclose(fid); insplot(avp(:,1:16));
fid=fopen('kf.bin'); xkpk=fread(fid, [31,inf], 'double')'; fclose(fid); kfplot(xkpk(:,[1:15,16:30,end]));
fid=fopen('rk.bin'); rk=fread(fid, [6,inf], 'double')'; fclose(fid); figure, plot(sqrt(rk(:,1:3))); xygo('rk / m/s');
return

glvs
dd = load('data1.txt');
tt = dd(:,1); ts=0.01; 
imu = [[dd(:,2:4)*glv.dps,dd(:,5:7)]*ts,tt]; imuplot(imu)
avp0 = [dd(:,8:10)*glv.deg,dd(:,11:13),dd(:,14:15)*glv.deg,dd(:,16),tt]; insplot(avp0);
gps = [dd(:,17:19),dd(:,20:21)*glv.deg,dd(:,22),tt]; gps = gps(gps(:,4)>0,:); gpsplot(gps);
magplot([dd(:,26:28),tt]);
baroplot([dd(:,29),tt], gps(:,[end-1,end]));

