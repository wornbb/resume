$fn = 30;
thou2mm = 0.0254;
translate([0,10]) cube([62,40,3]);
// probably radius should be 180
translate([129*thou2mm,805*thou2mm]) scale([thou2mm,thou2mm,1])cylinder(h=15, r1=132/2, r2=132/2); 



translate([2292*thou2mm,1605*thou2mm]) scale([thou2mm,thou2mm,1])cylinder(h=15, r1=132/2, r2=132/2); 

scale([thou2mm,thou2mm,1])cylinder(h=15, r1=132/2, r2=132/2); 

scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132); 
difference() {
    translate([15,0]) scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132); 
    
translate([15,0]) scale([thou2mm,thou2mm,1])cylinder(h=15, r1=132/2, r2=132/2); 


}
 