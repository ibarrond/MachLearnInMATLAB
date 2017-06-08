function out = outputTests(partId)
  % Random Test Cases
  X = reshape(3 * sin(1:1:30), 3, 10);
  Xm = reshape(sin(1:32), 16, 2) / 5;
  ym = 1 + mod(1:16,4)';
  t1 = sin(reshape(1:2:24, 4, 3));
  t2 = cos(reshape(1:2:40, 4, 5));
  t  = [t1(:) ; t2(:)];
  if partId == '1'
    [J] = nnCostFunction(t, 2, 4, 4, Xm, ym, 0);
    out = sprintf('%0.5f ', J);
  elseif partId == '2'
    [J] = nnCostFunction(t, 2, 4, 4, Xm, ym, 1.5);
    out = sprintf('%0.5f ', J);
  elseif partId == '3'
    out = sprintf('%0.5f ', sigmoidGradient(X));
  elseif partId == '4'
    [J, grad] = nnCostFunction(t, 2, 4, 4, Xm, ym, 0);
    out = sprintf('%0.5f ', J);
    out = [out sprintf('%0.5f ', grad)];
  elseif partId == '5'
    [J, grad] = nnCostFunction(t, 2, 4, 4, Xm, ym, 1.5);
    out = sprintf('%0.5f ', J);
    out = [out sprintf('%0.5f ', grad)];
  end 
end