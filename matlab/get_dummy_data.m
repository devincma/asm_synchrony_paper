function [var1, var2, var3, var4, var5, var6, var7] = get_dummy_data()
    var1 = rand(1, 10);       % Random numbers between 0 and 1, 1x10 array
    var2 = randi(100, 1, 5);  % Random integers between 1 and 100, 1x5 array
    var3 = linspace(0, 1, 20); % Linearly spaced numbers between 0 and 1, 1x20 array
    var4 = 'Dummy text';      % String variable
    var5 = true;              % Boolean variable
    var6 = eye(3);            % 3x3 identity matrix
    var7 = struct('field1', 'value1', 'field2', 2); % Structure variable
end