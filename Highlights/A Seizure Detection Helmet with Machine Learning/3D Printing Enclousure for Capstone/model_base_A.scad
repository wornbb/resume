$fn = 30;
thou2mm = 0.0254;
cube([62,62,1.5]);
translate([-2.5,-2.5,0]){
    difference(){
        cube([67,67,25]);
        translate([1.5,1.5,1.5]){
            cube([64,64,23.5]);}
        translate([7,0,5]){
            cube([50,70,20]);}
        translate([0,5,15]){
            cube([70,10,10]);}
        translate([0,40,15]){
            cube([70,10,10]);}
    }
}
// probably radius should be 180
translate([129*thou2mm,805*thou2mm]) {
scale([thou2mm,thou2mm,1])cylinder(h=17, r1=40, r2=40); 
scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132); 
}
translate([129*thou2mm,1605*thou2mm]){
  scale([thou2mm,thou2mm,1])cylinder(h=17, r1=40, r2=40); 
scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132);  
}
translate([2292*thou2mm,805*thou2mm]){ scale([thou2mm,thou2mm,1])cylinder(h=17, r1=40, r2=40); 
scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132);
}
translate([2292*thou2mm,1605*thou2mm]){ 
scale([thou2mm,thou2mm,1])cylinder(h=17, r1=40, r2=40); 
scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132);  }

