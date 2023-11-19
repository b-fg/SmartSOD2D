//ymin = 0.2
//utau = 0.03259259259259259
//mu = 0.0022222222222222222
//Deltaz+ 15.28
//Deltax+ 15.28
//Deltay+max 14.51
//y+ 1.348
//DOFs 40,543,300

d0 = 1.0;
Lx = 1100*d0;
Ly = 120*d0;
Lz = 125*d0;
s = d0;

//Main box
Point(1) = {-100, 0, 0, s};
Point(2) = {-100, Ly, 0, s};

Line(1) = {2, 1};

Transfinite Curve {1} = 80 Using Progression 0.970; // y

// Change layer to increase x subdivision
Extrude {Lx, 0, 0} { Line{1}; Layers{264*d0}; Recombine;}

// Change layer to increase z subdivision
Extrude {0, 0, Lz} { Surface{5}; Layers{30*d0}; Recombine;}

Physical Surface("wall") = {18}; //1
Physical Surface("sym") = {26};
Physical Surface("in") = {14};
Physical Surface("out") = {22};
Physical Surface("Periodic") = {5,27};
Physical Volume("fluid") = {1};

Mesh.MshFileVersion = 2.2;

Mesh.ElementOrder = 4;

Mesh 3;

Periodic Surface {27} = {5} Translate {0, 0, Lz};
