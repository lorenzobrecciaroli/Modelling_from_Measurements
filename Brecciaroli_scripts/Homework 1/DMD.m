clear all
clc
close all

addpath('optdmd-master/optdmd-master/src');
addpath('optdmd-master/optdmd-master/examples');

%% Definition of optdmd options
maxiter = 40; % maximum number of iterations
tol = 1.0e-6; % tolerance of fit
eps_stall = 1.0e-12; % tolerance for detecting a stalled optimization
opts = varpro_opts('maxiter',maxiter,'tol',tol,'eps_stall',eps_stall);

grey = [0.5 0.5 0.5];

%% Dataset
x1 = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 ...
      137 18 22 52 83 18 10 9 65];
x2 = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 ...
      18 37 50 35 12 12 25];

X = [x1; x2];


tend = length(x1)-1;
n = tend+1;
dt = 2; t1 = 1845; t2 = 1903;
t = t1:dt:t2;

figure; 
hold on; grid on;
plot(t,x1,'-*','DisplayName','Hare')
plot(t,x2,'-o','DisplayName','Lynx')
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')

%% Base opt-DMD
r = 2;
% 1 --- fit to unprojected data
imode = 1;
[w,e,b] = optdmd(X,t,r,imode);

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp(' (1.1) DMD (classic)')
disp('- Eigenvectors (w): ');
disp(w)
disp('- Weights (b): ');
disp(b)
disp('- Eigenvalues (e): ');
disp(e)

% reconstructed values
X_rec = w*diag(b)*exp(e*t);
relerr_r = norm(X_rec-X,'fro')/norm(X,'fro');

fprintf('example 1 --- fitting unprojected data\n')
fprintf('relative error in reconstruction %e\n',relerr_r)

figure; 
subplot(1,2,1)
hold on; grid on;
plot(t,x1,'-*','color',grey,'DisplayName','Hare')
plot(t,X_rec(1,:),'-*','DisplayName','Reconstructed Hare')
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')

subplot(1,2,2)
hold on; grid on;
plot(t,x2,'-o','color',grey,'DisplayName','Lynx')
plot(t,X_rec(2,:),'-*','DisplayName','Reconstructed Lynx')
legend('show');
xlabel('time [year]');


%% Time-delay opt-DMD
PLOT = 1;
gray_color = gray;
delay = [6];

for jj = 1:length(delay)
    H = [];
    for k = 1:delay(jj)
       H = [H; x1(k:end - delay(jj) + k); x2(k:end - delay(jj) + k)]; 
    end   
    clear t2 t_H
    dt = 2; t1 = 1845; t2 = 1903-(delay(jj)-1)*dt;
    t_H = t1:dt:t2;
    
    log.t_H{jj} = t;
    log.H{jj}   = H;
    
    % SVD Henkel Matrix
    clear u_H s_H v_H
    [u_H,s_H,v_H] = svd(H,'econ');

    if PLOT
        figure;
        plot(diag(s_H)/(sum(diag(s_H))),'o');
        title(['SVD: principal components - delay = ' num2str(delay(jj))]);
    end

    % DMD
    clear r
    r = 2*delay(jj);
    imode_init = 1;

    lbc_init = [-Inf*ones(r,1); -Inf*ones(r,1)];
    ubc_init = [zeros(r,1); Inf*ones(r,1)];
    copts_init = varpro_lsqlinopts('lbc',lbc_init,'ubc',ubc_init);
    
    clear w e b
    [w,e,b] = optdmd(H,t_H,r,imode_init,opts,[],[],copts_init);
    
    % Data reconstruction 
    clear X_rec
    X_rec = w*diag(b)*exp(e*(t));
    
    log.w{jj}   = w;
    log.e{jj}   = e;
    log.b{jj}   = b;
    log.X_rec{jj}   = X_rec;   

end

% PLOT
for jj = 1:length(delay)
    if PLOT
        if jj == 1
        figure; 
        axis_plot_rec(1) = subplot(1,2,1); hold on; grid on; box on; legend('show'); xlabel('time [year]'); ylabel('head numbers [thousands]')
        axis_plot_rec(2) = subplot(1,2,2); hold on; grid on; box on; legend('show'); xlabel('time [year]');
        plot(axis_plot_rec(1),t,x1,'-*','color',gray_color(100,:),'LineWidth',1,'DisplayName','Hare')
        plot(axis_plot_rec(2),t,x2,'-*','color',gray_color(100,:),'LineWidth',1,'DisplayName','Lynx')
        end
       
        plot(axis_plot_rec(1),log.t_H{jj},log.X_rec{jj}(1,:),'-*','DisplayName',['Reconstructed Hare. - delay = ' num2str(delay(jj))])
        plot(axis_plot_rec(2),log.t_H{jj},log.X_rec{jj}(2,:),'-*','DisplayName',['Reconstructed Lynx - delay = ' num2str(delay(jj))])
        
    end
    
    if PLOT
        if jj == 1
        figure; 
        axis_plot_eig = axes; 
        hold on; grid on; box on; legend('show'); title('eigenvalues');
        end
        scatter(axis_plot_eig,real(log.e{jj}),imag(log.e{jj}),20*[1:delay(jj)*2],'filled','DisplayName',['delay = ' num2str(delay(jj))]);
        
    end    
end


%% Time-delay opt-DMD with bagging 
delay = 6;
H = [];
for k = 1:delay

   H = [H; x1(k:end - delay + k); x2(k:end - delay + k)]; 
end   
dt = 2; t1 = 1845; t2 = 1903-(delay-1)*dt;
t_H = t1:dt:t2;

PLOT_ALL_BAGS = 1;
bag_size = 22;

% Random selection of samples
num_bags = 80;      
for kk = 1:num_bags
    index = randperm(length(X)-delay+1,bag_size);
    index = sort(index);
    index_set{kk} = index;
end


r = 2*delay;
imode = 1;

lbc = [-Inf*ones(r,1); -Inf*ones(r,1)];
ubc = [zeros(r,1); Inf*ones(r,1)];
copts = varpro_lsqlinopts('lbc',lbc,'ubc',ubc);

% Bagging main cycle
for jj = 1:num_bags
    t_bag = t_H(index_set{jj});
    H_bag = H(:,index_set{jj});
    [w_bag{jj},e_bag{jj},b_bag{jj},~,FLAG{jj}] = optdmd(H_bag,t_bag,r,imode,opts,[],[],copts);
end

% Reorder elements of bags for average
autoval = cell2mat(e_bag);

e_bag_sorted{1} = e_bag{1};
w_bag_sorted{1} = w_bag{1};
b_bag_sorted{1} = b_bag{1};
for kk=1:size(autoval,2)
    [~,ind] = sort(abs(autoval(:,kk)));
    e_bag_sorted{kk} = e_bag{kk}(ind);
    w_bag_sorted{kk} = w_bag{kk}(ind,:);
    b_bag_sorted{kk} = b_bag{kk}(ind);
end

autoval2 = cell2mat(e_bag_sorted);

figure; 
scatter(real(autoval2),imag(autoval2))
    
% Compute average
w_bag_vect = w_bag_sorted{1};
e_bag_vect = e_bag_sorted{1};
b_bag_vect = b_bag_sorted{1};
for m = 2:length(w_bag) % numero di bag testate
    w_bag_vect = w_bag_vect + w_bag_sorted{m};
    e_bag_vect = e_bag_vect + e_bag_sorted{m};
    b_bag_vect = b_bag_vect + b_bag_sorted{m};
end
w_bag_mean = w_bag_vect/length(w_bag);
e_bag_mean = e_bag_vect/length(w_bag);
b_bag_mean = b_bag_vect/length(w_bag);

% Compute std of eig distribution, remove small eigenvalues
e_bag_std = std(cell2mat(e_bag_sorted)');


% Reconstruct signal
dt = 2; t1 = 1845; t2 = 1903;
t = t1:dt:t2;
X_rec_H = w_bag_mean*diag(b_bag_mean)*exp(e_bag_mean*t);
X_rec_H = X_rec_H(1:2,:);

relerr_X_rec_H = norm(X_rec_H-X,'fro')/norm(X,'fro');

figure;
ax1 = subplot(1,2,1);
hold on; grid on;
ax2 = subplot(1,2,2);
hold on; grid on;
X_av_bag = zeros(2,length(t));
for jj =1:length(w_bag)
    X_rec_bag{jj} = w_bag{jj}*diag(b_bag{jj})*exp(e_bag{jj}*t);
    X_rec_bag{jj} = X_rec_bag{jj}(1:2,:);
    
    X_av_bag =  X_av_bag + X_rec_bag{jj};

    relerr_X_rec_bag{jj} = norm(X_rec_bag{jj}-X,'fro')/norm(X_rec_bag{jj},'fro');

    plot(ax1,t,X_rec_bag{jj}(1,:),'Color',[0.8,0.8,0.8],'HandleVisibility','off')
    plot(ax2,t,X_rec_bag{jj}(2,:),'Color',[0.8,0.8,0.8],'HandleVisibility','off')
end
X_av_bag = X_av_bag./length(w_bag);

subplot(1,2,1)
plot(ax1,t,x1,'-*','color','b','DisplayName','Hare','MarkerSize',4)
plot(ax1,t,X_av_bag(1,:),'-*','Color','r','DisplayName','Avg Hare','MarkerSize',4)
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')
ylim([-50,150]);

subplot(1,2,2)
plot(t,x2,'-*','color','b','DisplayName','Lynx','MarkerSize',4)
plot(ax2,t,X_av_bag(2,:),'-*','Color','r','DisplayName','Avg Lynx','MarkerSize',4)
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')
ylim([-50,100]);



