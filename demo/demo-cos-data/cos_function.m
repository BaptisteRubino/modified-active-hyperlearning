function [ result ] = cos_function( h )
% function used as a test function in the Profit implementation
% INPUT
% h -> linespace from 0 to 1
% OUTPUT
% result -> array of points

result = h + cos(10*h);

end

