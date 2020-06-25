t = (1/16:1/8:1)'*2*pi;
x = sin(t);
y = cos(t);
y2 = sin(t)



fill(x,y,'r')
hold on
fill(x,y2,'b')
title('yo')
axis square

[~,h_legend] = legend('area1','area2');
PatchInLegend = findobj(h_legend, 'type', 'patch');

