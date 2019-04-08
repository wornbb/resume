$fn = 30;
thou2mm = 0.0254;

translate([1.5+1.5,1.5+1.5,1.5]){
translate([129*thou2mm,805*thou2mm]) {
    difference(){
    scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132);
    scale([thou2mm,thou2mm,1])cylinder(h=17, r1=60, r2=60); 

    } 
}
translate([129*thou2mm,1605*thou2mm]){
    difference(){
    scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132);
    scale([thou2mm,thou2mm,1])cylinder(h=17, r1=60, r2=60); 

    }  
}
translate([2292*thou2mm,805*thou2mm]){ 
    difference(){
    scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132);
    scale([thou2mm,thou2mm,1])cylinder(h=17, r1=60, r2=60); 

    }
}
translate([2292*thou2mm,1605*thou2mm]){ 
    difference(){
    scale([thou2mm,thou2mm,1])cylinder(h=10, r1=132, r2=132);  
    scale([thou2mm,thou2mm,1])cylinder(h=17, r1=60, r2=60); }
    }
}

translate([67,0,25]){
rotate([0,180,0]){
translate([0,0,0]){
    difference(){
        cube([67,67,25]);
        translate([1.5,-1,0]){
            cube([64,70,23.5]);}
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
}
}
}