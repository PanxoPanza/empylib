// Plantailla Gmsh para un nanohilo cilíndrico de TiO2
    DefineConstant[ R = 0.35 ]; // radio del nanohilo
    DefineConstant[ H = 7.0 ]; // longitud del nanohilo
    DefineConstant[ N = 4.0 ]; // subdivisiones en el radio
    DefineConstant[ Smooth = 0 ]; // 1 si se desean extremos redondeados
    DefineConstant[ Hscale = 10 ]; // escala de mallado axial (relativa a L)
    
    L = Pi*R/(2*N);
    NH = Ceil[ Hscale*R/L ];
    NR = Ceil[ R/L ];
    
    If (Smooth == 0)
      Point(0) = { 0, 0, 0.5*H };
      Extrude{ R, 0, 0 } { Point{0}; Layers{NR}; }
    EndIf
    
    If (Smooth == 1)
      Point(0) = { 0, 0, 0.5*H + R };
      Extrude{ {0,1,0}, {0,0,0.5*H}, -Pi/2 } { Point{0}; Layers{N}; }
    EndIf
    
    Extrude{ 0, 0, -H } { Point{1}; Layers{NH}; }
    
    If (Smooth == 0)
      Extrude{ -R, 0, 0 } { Point{2}; Layers{NR}; }
    EndIf
    
    If (Smooth == 1)
      Point(4) = { 0, 0, -0.5*H - R };
      Extrude{ {0,1,0}, {0,0,-0.5*H}, -Pi/2 } { Point{3}; Layers{N}; }
    EndIf
    
    Extrude{ {0,0,1}, {0,0,0}, Pi/2 } { Line{1,2,3}; Layers{N}; }
    Extrude{ {0,0,1}, {0,0,0}, Pi/2 } { Line{4,7,11}; Layers{N}; }
    Extrude{ {0,0,1}, {0,0,0}, Pi/2 } { Line{14,17,21}; Layers{N}; }
    Extrude{ {0,0,1}, {0,0,0}, Pi/2 } { Line{24,27,31}; Layers{N}; }
    