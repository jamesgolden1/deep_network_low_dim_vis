function E = latticenet()
% Visualize how a deep network in low dimensions warps the input data
% space in order to correctly classify two datasets. Based on code from
% Olshausen, VS265, 2010.

% "Trained networks... contract space at the center of decision volumes and
% expand space in the vicinity of decision boundaries" - Nayebi & Ganguli,
% 2017.
% 
% y = sig(Wx), z = sig(Vy) undergoing supervised training to separate blue
% and red, adapted from Olshausen, 2010. Inspired by Fig 6 of Olshausen &
% Field 2005 (http://www.rctn.org/bruno/CTBP/olshausen-field05.pdf) and
% Olah 2014 (http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/).

% In order to correctly classify these data points in a 2D space as red or
% blue, a supervised two-layer network needs some number of feature vectors
% in the first layer (W). If the projections onto those three feature
% vectors are plotted in a representation space where the features are
% orthogonal, we can observe how the network learns to warp the 2D input
% space lattice in 3D representation space in order to make classification
% possible with a single plane (blue cyan points), where projection onto a
% vector normal to the cyan plane separates the two classes (rightmost
% panel).
% 
% Learning feature vectors in W is equivalent to warping the input space
% within the representation space, and learning feature vector V is the
% placement of the separating plane (cyan points). Convolutional networks
% do something like this in high dimensions to classify images, and
% measurements of the local curvature of the input space reveal how
% individual neurons and layers contribute to successful recognition.
% 
% The network is described by these equations for the two layers: y =
% sig(Wx), z = sig(Vy), where sig is a sigmoid nonlinearity. The data is
% limited to the (0,1) cube by the sigmoid function.
% 
% When the classes are randomized, a deeper network with two hidden layers
% of 16 units (projected back into a penultimate layer of 3 units) is
% almost powerful enough to "memorize" a dataset by extreme warping of the
% input data space. Note that at the end a number of points are still
% classified incorrectly.

% The "memorize" case is included as a curiousity - note that without a
% validation set, the network is not really learning the data class
% distribution. However, the framework is flexible enough that the network
% can indeed learn this arbitrary separation of classes, with extreme
% warping of the input space.

% Setting the 'randomTeacherFlag' will randomize the class and allow for
% testing if the network can "memorize" the arbitrary distribution.

% The network is initialized with pseudo-identity matrices in order to show
% the 2D data space as a square lattice initially in the representation space
% of the penultimate layer, then for the first 500 frames it is shifted to
% a random initiliazation, which distorts the lattice. Learning begins
% when the cyan classification plane appears.

% James Golden
% jgolden1 ~at~ stanford.edu

%% Choose data and network parameters

% paramVals = 1; % 2D data, linearly separable, 2 layer network, 
% paramVals = 2; % 2D data, not linearly separable, 2 layer network
% paramVals = 3; % 2D data, not linearly separable, three layer network
paramVals = 4; % 2D data, not linearly separable, six layer network, "memorize"


%% Set parameters, choose data class layout

switch paramVals
    case 1
        rng(1585775);
        
        % Set number of layers, number of units per layer, of network
        w_weights = [2 2];        
        
        % Set localized class data
        randomTeacherFlag = 0;

        % View for penultimate layer figure
        az = 0; el = 90;
        
        % learning rate
        eta0 = .1;
        
        num_trials = 1000;
        
        % Get 2D dataset with two classes, linearly separable
        [data, teacher, N, K] = buildData2d(randomTeacherFlag);
        
    case 2
        rng(2590825);
        
        % Set number of layers, number of units per layer, of network
        w_weights = [2 3];
        
        % Set localized class data
        randomTeacherFlag = 0;
        az = -46; el = 8;

        eta0 = .15;
        num_trials = 1400;      
        
        % Get 2D dataset with two classes, not linearly separable
        [data, teacher, N, K] = buildData(randomTeacherFlag);
        % rng(23513);
    case 3
        
        rng(1585775);
        
        % Set number of layers, number of units per layer, of network
        w_weights = [2 8 8 3];
        
        % Set localized class data
        randomTeacherFlag = 0;

        az = -46.5+18; el = 25.5-18;
        
        eta0 = .1;
        num_trials = 1400;
        
        % Get 2D dataset with two classes
        [data, teacher, N, K] = buildData(randomTeacherFlag);

    case 4        
        rng(2958);
        
        % Set number of layers, number of units per layer, of network
        w_weights = [2 64 64 64 3];
        
        % Set random class data, "memorize"
        randomTeacherFlag = 1;
        az = 68; el = 24;
        eta0 = .15;
        num_trials = 10000;
        
        % Get 2D dataset with two classes
        [data, teacher, N, K] = buildData(randomTeacherFlag);
end


%% Set classes for dataset

% Set classes for supervised learning
half1 = teacher>.5;
half2 = teacher<.5;

% Set nonlinear function between layers
sigmoid = @(x) 1 ./ (1 + exp(-1*x));

% Set learning rate
etaarr=eta0*ones(30*460,1);
num_rep = 24;
etaarr = [etaarr; eta0*ones(num_rep*460,1)]/3.05;
 
% Set parameters for making the gif
giffac = 5;
gifmod = giffac*round([40/giffac:(120-40)/(giffac*10000):120/giffac]);

%% Build deep network

[w, w0, wR] = buildDeepNet(w_weights);

%% Transform data points with deep network

datay = transformData(sigmoid,w,w0,data);

%% Plotting parameters

% Number of frames after learning before gif repeats
endframes = 20;

% Lattice spacing
divfac = .125/2;
maxgr = 8/1; 
maxst = -3/divfac;

% Decision plane (cyan points) spacing
pxyint = .05;
[pmeshx,pmeshy] = meshgrid(0:pxyint:1,0:pxyint:1);
pxy = [pmeshx(:)'; pmeshy(:)'];
[xin,yin] = meshgrid(maxst+[1:maxgr/divfac],maxst+[1:maxgr/divfac]);
grid_x = divfac*[xin(:)'; yin(:)'];

%% Initialize figure
 
hf0 = figure(1);

set(gcf,'position',[1         410        1440         388]);
  
clf;

h11 = subplot(131);
h22 = subplot(132);
h33 = subplot(133);

h11 = buildPlot1(h11,data,half1,half2);

h22 = buildPlot2(h22,sigmoid,datay,w,w0,az,el,half1,half2,xin,yin,pxy,0);

z = datay{length(w)+1};
h33 = buildPlot3(h33,z,half1,half2);


%% Initialize GIF

name_str = ['n_layer_warping_' num2str(cputime*100) '.gif'];
path_str ='/Users/james/Documents/MATLAB/lab2-supervised_learning/';
gifname = [path_str name_str];

gifinit(hf0, gifname);


%% Start with pseudo-identity matrix for each layer for close to square lattice in representation space

for layer = 1:length(w_weights)-1        
    dwR{layer} = w{layer} - wR{layer};
end


%% Train network

for t=[1:1*num_trials]
    
    if t < 500        
        for layer = 1:length(w_weights)-1        
             w{layer} = w{layer} + (1/500)*dwR{layer};
        end
        
        
        datay{1} = data;
        datay = transformData(sigmoid,w,w0,datay{1});
        
    else
        
        eta = etaarr(t);
        
        
        % compute new representation        
        datay{1} = data;
        datay = transformData(sigmoid,w,w0,datay{1});

        % get class predictions
        layer = length(w_weights);
        z = datay{length(w_weights)+1};
        
        % compute error
        error = 1/2*(teacher-z).^2;
        
        % compute gradient at ouptut layer
        delta_y{layer+1} = (teacher-z).*((z).*(1-z));
        
        % backpropagate gradient to lower layers
        for layer = length(w_weights):-1:1%-1
            dsig_y{layer} = datay{layer}.*(1-datay{layer});
            delta_y{layer} = (w{layer}*delta_y{layer+1}).*dsig_y{layer};
        end
        
        % update weights
        for layer = 1:length(w_weights)%-1
            
            dw{layer} = eta*datay{layer}*delta_y{layer+1}';
            w{layer} = w{layer} + dw{layer};
            
            dw0{layer} = sum(eta*delta_y{layer+1}')';
            w0{layer} = w0{layer} + dw0{layer};
        end
        
        % store error
        E(t) = sum(abs(error));
    end
    
    % end
    
    %% Update figures in gif
    if ~mod(t,gifmod(t))
        h22 = buildPlot2(h22,sigmoid,datay,w,w0,az,el,half1,half2,xin,yin,pxy,t);
        
        z = datay{length(w)+1};
        h33 = buildPlot3(h33,z,half1,half2);
        
        gifwrite(hf0, gifname)        
    end
end

% Include some end frames so gif doesn't immediately restart
for t = 1:endframes
    gifwrite(hf0, gifname);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [data, teacher, N, K] = buildData(randomTeacherFlag)

radlim = 0.5;
radlimbig = 3.5;
datacenter = randn(2,200);

databig = datacenter(:,(sqrt(datacenter(1,:).^2 + datacenter(2,:).^2) > radlim +.2 ) & (sqrt(datacenter(1,:).^2 + datacenter(2,:).^2) < radlimbig));

datacenter = randn(2,2000);

% datasmall = datacenter(:,abs(datacenter(1,:)<0.2) & abs(datacenter(1,:)<0.2));

datasmall = datacenter(:,sqrt(datacenter(1,:).^2 + datacenter(2,:).^2) < radlim - .1);
clear datacenter
data = .5+[databig(:,1:50) datasmall(:,1:50)];

[N K]=size(data);
teacher=[ ones(1,K/2) zeros(1,K/2)];

if randomTeacherFlag
    teacher = round(rand(size(teacher)));
end


%%
function [data, teacher, N, K] = buildData2d(randomTeacherFlag)

databig1 = 1*rand(1,100);
databig2 = .35+.1*rand(1,100);
databig = [databig1; databig2];
% databigInd = databig(1,:)>.5 & databig(1,:)<.7;

datasmall1 = 1*rand(1,100);
datasmall2 = .55+.1*rand(1,100);
datasmall = [datasmall1; datasmall2];
% datasmallInd = datasmall(1,:)<.5 & datasmall(1,:)>.3;
% datasmall(1,1:50) = datasmall(1,1:50) + .5; 
data = [databig(:,(1:50)) datasmall(:,(1:50))];
size(data)
[N K]=size(data);
teacher=[ ones(1,K/2) zeros(1,K/2)];

if randomTeacherFlag
    teacher = round(rand(size(teacher)));
end
%%

function [w, w0, wR] = buildDeepNet(w_weights)
 
for layer = 1:length(w_weights)-1
    w{layer} = zeros(w_weights(layer),w_weights(layer+1)); 
    
    for layeri = 1:min(w_weights(layer),w_weights(layer+1))
        w{layer}(layeri,layeri) = .1;
    end
    wR{layer} = randn(w_weights(layer),w_weights(layer+1)); 
%     w{layer} = w{layer} + .01*randn(size(w{layer}));
    w0{layer} = .1*randn(w_weights(layer+1),1); 
    
end


w{layer+1}=randn(w_weights(layer+1),1); % second layer weights
w0{layer+1}=randn(1);

%%
function datay = transformData(sigmoid,w,w0,data)

datay{1} = data;
for layer = 1:length(w)%-1
    datay{layer+1}=sigmoid(w{layer}'*datay{layer} + w0{layer}*ones(1,size(datay{layer},2)));

end

%%


function h11 = buildPlot1(h11,data,half1,half2)
delete(h11);
h11 = subplot(131);

hold on;
scatter(data(1,half1),data(2,half1),180,'bo');%,'filled')
scatter(data(1,half2),data(2,half2),180, 'ro');%,'filled')

hold on

% axis image, 
% axis([-3 3 -3 3])
grid on;

title('Input space');

set(gca,'fontname','CMU Serif');

set(gca,'fontsize',14);

axis([min(data(1,:)) max(data(1,:)) min(data(2,:)) max(data(2,:))]);
   

%%

function h22 = buildPlot2(h22,sigmoid,datay,w,w0,az,el,half1,half2,xin,yin,pxy,t)

% initialize hidden units plot
delete(h22);
% cla;
h22 = subplot(132);
hold on;
grid on;

divfac = .125/1;
maxgr = 8/1; 
maxgrm1 = maxgr/divfac;
maxst = -3/divfac;
c1 = 'k'; c2 = 'k'; c3 = 'k';
shiftfac = .125;

hlctr = 0; addvctr = 0;
addvall = [0:shiftfac:1-shiftfac ];

addv = 0;

y2 = datay{length(w)};
v = w{length(w)}; v0 = w0{length(w)};

if size(y2,1) < 3
    y2(3,:) = 0;
end

hold on;
hy3d(1)=scatter3(y2(1,half1),y2(2,half1),y2(3,half1),180,'bo','filled');%,'markersize',30);
hold on
hy3d(2)=scatter3(y2(1,half2),y2(2,half2),y2(3,half2),180,'ro','filled');%,'markersize',30);

% [xin,yin] = meshgrid(maxst+[1:maxgr/divfac],maxst+[1:maxgr/divfac]);
grid_x = divfac*[xin(:)'; yin(:)'];


datay_grid{1} = grid_x;
for layer = 1:length(w)%-1
    datay_grid{layer+1}=sigmoid(w{layer}'*datay_grid{layer} + w0{layer}*ones(1,size(datay_grid{layer},2)));
end
layer = layer-1;

gs = (size(xin,1));

if size(datay_grid{layer+1},1) < 3
    datay_grid{layer+1}(3,:) = 0;
end

for ri = 1:gs    
    plot3(datay_grid{layer+1}(1,(ri-1)*gs+1:ri*gs),datay_grid{layer+1}(2,(ri-1)*gs+1:ri*gs),datay_grid{layer+1}(3,(ri-1)*gs+1:ri*gs),'k');    
end;

%
% [xin,yin] = meshgrid(maxst+[1:maxgr/divfac],maxst+[1:maxgr/divfac]);

grid_x = divfac*[yin(:)'; xin(:)'];

datay_grid{1} = grid_x;
for layer = 1:length(w)%-1
    datay_grid{layer+1}=sigmoid(w{layer}'*datay_grid{layer} + w0{layer}*ones(1,size(datay_grid{layer},2)));    
end


layer = layer-1;
if size(datay_grid{layer+1},1) < 3
    datay_grid{layer+1}(3,:) = 0;
end
for ri = 1:gs       
    plot3(datay_grid{layer+1}(1,(ri-1)*gs+1:ri*gs),datay_grid{layer+1}(2,(ri-1)*gs+1:ri*gs),datay_grid{layer+1}(3,(ri-1)*gs+1:ri*gs),'k');
end;


grid on;
% axis equal
view(az,el);
% view(-46.5+18-t/8,25.5-18+t/8);

ax1 = axis;
% axis equal
fgint = 20;
set(gca,'fontname','CMU Serif');

set(gca,'fontsize',14);

% z = datay{layer+2};


% z0 = v;% + v0;
% z1 = -1*v ;%+ v0
% z0 = [0;0;0]; z1 = -v/(1*v0); 
layer = layer-1;
% if size(datay_grid{length(datay_grid)-1},1) == 3
if sum(abs(datay_grid{length(datay_grid)-1}(3,:))) ~= 0

pz = (-v0-(v(1:2)'*pxy))/v(3);

scatter3(pxy(1,:),pxy(2,:),pz,'filled','c');

zlabel('jamesgolden.net');
else
pz = (-v0-(v(1)'*pxy(1,:)))/v(2);

scatter(pxy(1,:),pz,'filled','c');

ylabel('jamesgolden.net');
end

title('Representation space - penultimate layer');
axis(ax1);

hh1 = findobj(gca,'Type','Scatter');
hh2 = findobj(gca,'Type','line');
set(gca,'children',[hh1; hh2]);
    
drawnow;

%%

function h33 = buildPlot3(h33,z,half1,half2)
delete(h33);
h33 = subplot(133); hold on;
scatter(zeros(size(z(1,half1))),z(1,half1),180,'bo')%,'filled')
scatter(ones(size(z(1,half2))),z(1,half2),180,'ro')%,'filled')
% scatter(data(1,half2),data(2,half2),180, 'ro','filled')
plot([-2 2],[.5 .5],'c','linewidth',4);

grid on;
title('Representation space - output layer');
set(gca,'fontname','CMU Serif');

set(gca,'fontsize',14);

axis([-.5 1.5 -.1 1.1]);
axis equal

%%
function gifinit(hf0, fname)

% Capture the plot as an image 
frame = getframe(hf0); 
im = frame2im(frame); 
[imind,cm] = rgb2ind(im,256); 
% Write to the GIF File 
% if n == 1 
imwrite(imind,cm,fname,'gif', 'Loopcount',inf); 
% % else 
% imwrite(imind,cm,[path_str name_str],'gif','WriteMode','append','DelayTime',.05); 
% end 

function gifwrite(hf0, fname)

% Capture the plot as an image 
frame = getframe(hf0); 
im = frame2im(frame); 
[imind,cm] = rgb2ind(im,256); 
% Write to the GIF File 
% if n == 1 
% if n == 1 
% imwrite(imind,cm,[path_str name_str],'gif', 'Loopcount',inf); 
% % else 
imwrite(imind,cm,fname,'gif','WriteMode','append','DelayTime',.05); 