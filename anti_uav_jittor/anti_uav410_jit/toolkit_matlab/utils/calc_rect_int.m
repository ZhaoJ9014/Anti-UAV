function overlap = calc_rect_int(A, B)
% Calculate overlap of two rectangles
leftA   = A(:,1);
bottomA = A(:,2);
rightA  = leftA + A(:,3) - 1;
topA    = bottomA + A(:,4) - 1;

leftB   = B(:,1);
bottomB = B(:,2);
rightB  = leftB + B(:,3) - 1;
topB    = bottomB + B(:,4) - 1;

tmp     = (max(0, min(rightA, rightB) - max(leftA, leftB)+1 )) .* (max(0, min(topA, topB) - max(bottomA, bottomB)+1 ));
areaA   = A(:,3) .* A(:,4);
areaB   = B(:,3) .* B(:,4);
overlap = tmp./(areaA+areaB-tmp);
end