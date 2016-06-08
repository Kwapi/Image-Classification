function [isMode, mode ] = getMode( data )

% get unique values
unq = unique(data);

% count how many times they occur
valueCount = histc(data,unq);

% put values and their occurences together
combined = horzcat(unq,valueCount);

% sort ascending by the number of occurences
combined = sortrows(combined,2);

% the mode will be the last value (the biggest number of occurences)
mode = combined(end,1);

% if all the number of occurences are the same (ie: three completely
% different values) - then the mode cannot be calculated

if all(valueCount == valueCount(1))
    isMode = false;
else
    isMode = true;
end

end

