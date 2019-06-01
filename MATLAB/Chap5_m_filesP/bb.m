function X=bb(t)
W=bmotion(t); X=W-W(end)*(t-t(1))/(t(end)-t(1));

