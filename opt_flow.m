file_name = '37000_SPD16x16.mat';
data = load(file_name);
[cnt_rows, cnt_cols, cnt_meas] = size(data.sign_bb(:, :, :));
B = zeros(cnt_rows, cnt_cols, cnt_meas);
for k = 1:cnt_meas
    B(:, :, k) = rot90(data.sign_bb(:, :, k), 2);
end
dt = cell2mat(data.Data(1, 2)) * 1e-3;
t_s = cell2mat(data.Data(2, 2));


t_cons_s = 135;
t_cons_e = 175;

ticks = fix(5 / dt);

v = VideoWriter('flow.avi');
open(v);

opticFlow = opticalFlowHS;
h = figure;
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);
i = 0;
tick_start = fix((166 - t_s) / dt);
while i < ticks
    i = i + 1;
    frame = B(:, :, tick_start + i);
    labeledImage = bwlabel(true(size(frame)));
    props = regionprops(labeledImage, frame, 'WeightedCentroid');
    flow = estimateFlow(opticFlow, frame(1:14, 1:10));
    imagesc(frame)
    hold on
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',60 ,'Parent',hPlot);
    plot(props.WeightedCentroid(1), props.WeightedCentroid(2), '*');
    t = i * dt + 166;
    title([num2str(t), 'ms'])
    fr = getframe(h);
    writeVideo(v,fr);
    hold off
    pause(0.0000001)
end

close(v);

