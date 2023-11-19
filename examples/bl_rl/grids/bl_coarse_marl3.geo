// GEO file to generate the mesh of a TBL for MARL

//ymin = 0.2
//utau = 0.03259259259259259
//mu = 0.0022222222222222222
//Deltaz+ 38.2
//Deltax+ 38.2
//Deltay+max 25.08
//y+ 1.83

d0 = 1.0;
Lx = 1100*d0;
Ly = 120*d0;
Lz = 125*d0;
s = d0;
n_invars = 3;

//Main box
Point(1) = {-100, 0, 0, s};
Point(2) = {-100, Ly, 0, s};

Line(1) = {2, 1};

Transfinite Curve {1} = 50 Using Progression 0.947; // y

// Change layer to increase x subdivision
Extrude {Lx, 0, 0} { Line{1}; Layers{105*d0}; Recombine;}

// Change layer to increase z subdivision
vol1[] = Extrude {0, 0, Lz/n_invars} { Surface{5}      ; Layers{4*d0}; Recombine;};
vol2[] = Extrude {0, 0, Lz/n_invars} { Surface{vol1[0]}; Layers{4*d0}; Recombine;};
vol3[] = Extrude {0, 0, Lz/n_invars} { Surface{vol2[0]}; Layers{4*d0}; Recombine;};

Transfinite Volume{vol1[1],vol2[1],vol3[1]};
Recombine   Volume{vol1[1],vol2[1],vol3[1]};

Physical Surface("wall_inv1") = {vol1[3]};
Physical Surface("wall_inv2") = {vol2[3]};
Physical Surface("wall_inv3") = {vol3[3]};
Physical Surface("sym") = {vol1[5],vol2[5],vol3[5]};
Physical Surface("in") = {vol1[2],vol2[2],vol3[2]};
Physical Surface("out") = {vol1[4],vol2[4],vol3[4]};
Physical Surface("Periodic") = {5,71};
Physical Volume("fluid") = {vol1[1],vol2[1],vol3[1]};

Mesh.MshFileVersion = 2.2;

Mesh.ElementOrder = 4;

Mesh 3;

Periodic Surface {71} = {5} Translate {0, 0, Lz};
