f = figure('Visible','off','Position',[360,200,1000,600]);

f.Visible = 'on';
for i = 1:1000
    x = 1:1000;
    y = (1:1000)*i;
    plot(x,y);
    drawnow
end