% load fcmdata.dat
% 
% [centers,U] = fcm(fcmdata,3);
% 
% maxU = max(U);
% index1 = find(U(1,:) == maxU);
% index2 = find(U(2,:) == maxU);
% index3 = find(U(3,:) == maxU);
% 
% plot(fcmdata(index1,1),fcmdata(index1,2),'ob')
% hold on
% plot(fcmdata(index2,1),fcmdata(index2,2),'or')
% plot(centers(1,1),centers(1,2),'xb','MarkerSize',15,'LineWidth',3)
% plot(centers(2,1),centers(2,2),'xr','MarkerSize',15,'LineWidth',3)
% plot(centers(3,1),centers(3,2),'xr','MarkerSize',15,'LineWidth',3)
% hold off
% 
% 
