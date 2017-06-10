clear, clc, close all

vgg16.train_loss = [3.57373512077,3.2367178154,3.07826091909,2.96034431982,2.8601478653,2.76807180786,2.6825678792,2.62100021839,2.54931354094,2.48767989635];
vgg16.val_loss = [2.94154109001,2.90655622959,2.65562294483,2.54271100044,2.53772690773,2.44318650723,2.34101639748,2.29519654751,2.23758168936,2.29149703264];
vgg16.val_acc = [0.348999984264,0.370699987411,0.414399984479,0.427499977946,0.434299981594,0.455999979377,0.474099977612,0.480899973512,0.493499976397,0.482699980736]
vgg16.train_acc = [0.216359990954,0.273229985923,0.304179984331,0.327999985665,0.34974998489,0.368509983718,0.384609985173,0.398709984243,0.415199980915,0.42729998219]

vgg16.train_loss_fine= [2.44424158859,2.37689440393,2.32347535944,2.28133661628,2.23602770376,2.18165408349,2.1470337534,2.10625493097,2.06304118252,2.02640776896];
vgg16.val_loss_fine = [2.2145600915,2.18518424511,2.13096219063,2.11214374542,2.13086363077,2.09223309755,2.06098106384,2.04631062984,2.04768262863,2.02704145432];
vgg16.val_acc_fine = [0.494799978137,0.503699973822,0.515899965763,0.516799975634,0.511199970245,0.521799970865,0.530999972224,0.537499973178,0.527599965334,0.538799966574];
vgg16.train_acc_fine= [0.438839981318,0.454079980552,0.465349980354,0.473579978466,0.484859978855,0.49401997596,0.502219974041,0.512109970391,0.5209899683,0.529749967396];

vgg19.val_acc = [0.183899991512,0.223799989223,0.237499989569,0.251499988735,0.261799987853,0.258999989033,0.273399984837,0.272199987471,0.273399986625,0.272199988663];
vgg19.train_acc = [0.0662799980268,0.114859996542,0.131659995176,0.143789994404,0.150519994423,0.15482999371,0.159569993317,0.163179994047,0.162809994027,0.168439993486];
vgg19.val_loss = [3.77646868706,3.58751558781,3.53108965397,3.48501555443,3.46915997505,3.47512831211,3.47241164207,3.46945554733,3.48040792942,3.47856464386];
vgg19.train_loss = [4.6630418272,4.2060252161,4.07810920715,3.99909072351,3.94800776815,3.92675634098,3.91342787838,3.88350450087,3.88196943283,3.85984200811];

vgg19.val_loss_fine = [3.22469905376,2.95707035065,2.83038408756,2.77626327515,2.65663205624,2.56605816841,2.58488180637,2.47730269909,2.39109867096,2.35955016136,2.32096466541,2.3257585907,2.22151320219,2.27358263969,2.22759646893,2.21644414902,2.19179564953,2.17112591743,2.13289859295,2.09652055979];
vgg19.val_acc_fine = [0.301099982262,0.352899979949,0.373599983454,0.386999982595,0.410599978566,0.419499983191,0.420199976563,0.441299980283,0.456099982858,0.466799980402,0.468799982071,0.47379997611,0.488399972916,0.48339997828,0.489599974751,0.500699974298,0.498999971151,0.504999975562,0.510799973011,0.520099971294];
vgg19.train_acc_fine = [0.182039992988,0.236909989744,0.265499986947,0.291369985342,0.312299984694,0.328419983715,0.342649984062,0.357899983764,0.373399983764,0.386469985008,0.39676998353,0.409539981902,0.420289982378,0.432579982758,0.440689982533,0.451329980493,0.458579981089,0.469989978731,0.47966997838,0.486089976966];
vgg19.train_loss_fine = [3.73661379242,3.40710076904,3.25604561853,3.12938036919,3.03250509262,2.95156576252,2.88448387957,2.81070406199,2.7327624774,2.68027072859,2.62747524881,2.57128197289,2.52526789188,2.47381573248,2.43484450364,2.38896304774,2.35112531376,2.30228099513,2.26616183591,2.23069068742];

Colors = {'r', 'b'};

epoch = 1:length(vgg16.train_loss)
% figure, hold on, box on
% plot(epoch, vgg16.train_loss, '-o', 'Color', Colors{1}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
% plot(epoch, vgg16.val_loss, '--o', 'Color', Colors{1}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
% plot(epoch, vgg19.train_loss, '-o', 'Color', Colors{2}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
% plot(epoch, vgg19.val_loss, '--o', 'Color', Colors{2}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
% xlabel('Epoch', 'FontSize', 12)
% ylabel('Loss', 'FontSize', 12)
% legend('Modified VGG16: Training', 'Modified VGG16: Val', 'Modified VGG19: Training', 'Modified VGG19: Val')

epoch = 1:length(vgg16.train_acc)
figure, hold on, box on
plot(epoch, 100*vgg16.train_acc, '-o', 'Color', Colors{1}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
plot(epoch, 100*vgg16.val_acc, '--s', 'Color', Colors{1}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
plot(epoch, 100*vgg19.train_acc, '-o', 'Color', Colors{2}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
plot(epoch, 100*vgg19.val_acc, '--s', 'Color', Colors{2}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
xlabel('Epoch', 'FontSize', 12)
ylabel('Accuracy (%)', 'FontSize', 12)
legend('Modified VGG16: Training', 'Modified VGG16: Val', 'Modified VGG19: Training', 'Modified VGG19: Val', 'Location', 'SouthEast')
set(gca, 'FontSize', 12)
axis([1 10 5 50])
saveas(gca, 'course_acc_vgg16_vgg19', 'epsc')

% epoch = 1:length(vgg16.train_loss_fine)
% figure, hold on, box on
% plot(epoch, vgg16.train_loss_fine, '-o',  'Color', Colors{1}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
% plot(epoch, vgg16.val_loss_fine, '--o',  'Color', Colors{1}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
% plot(1:length(vgg19.train_acc_fine), vgg19.train_loss_fine, '-o',  'Color', Colors{2}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
% plot(1:length(vgg19.train_acc_fine), vgg19.val_loss_fine, '--o',  'Color', Colors{2}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
% xlabel('Epoch', 'FontSize', 12)
% ylabel('Loss', 'FontSize', 12)
% legend('Modified VGG16: Training', 'Modified VGG16: Val', 'Modified VGG19: Training', 'Modified VGG19: Val')

epoch = 11:20
figure, hold on, box on
plot(epoch, 100*vgg16.train_acc_fine, '-o', 'Color', Colors{1},  'LineWidth', 2, 'MarkerFaceColor', 'w')
plot(epoch, 100*vgg16.val_acc_fine, '--s', 'Color', Colors{1}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
plot(1:length(vgg19.train_acc_fine), 100*vgg19.train_acc_fine, '-o', 'Color', Colors{2},  'LineWidth', 2, 'MarkerFaceColor', 'w')
plot(1:length(vgg19.train_acc_fine), 100*vgg19.val_acc_fine, '--s', 'Color', Colors{2}, 'LineWidth', 2, 'MarkerFaceColor', 'w')
xlabel('Epoch', 'FontSize', 12)
ylabel('Accuracy (%)', 'FontSize', 12)
legend('Modified VGG16: Training', 'Modified VGG16: Val', 'Modified VGG19: Training', 'Modified VGG19: Val', 'Location', 'SouthEast')
set(gca, 'FontSize', 12)
axis([11 20 30 60])
saveas(gca, 'fine_acc_vgg16_vgg19', 'epsc')