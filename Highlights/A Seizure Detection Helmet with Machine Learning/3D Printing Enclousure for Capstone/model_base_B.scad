$fn = 30;
thou2mm = 0.0254;

difference(){
    
        union(){
//cube([62,62,1.5]);
//translate([-2.5,-2.5,0]){
    cube([5,5,23.5]);
    translate([62,0,0]){cube([5,5,23.5]);}
    translate([0,62,0]){cube([5,5,23.5]);}
    translate([62,62,0]){cube([5,5,23.5]);}
    difference(){
        cube([67,67,25]);
        translate([1.5,1.5,1.5]){
            cube([64,64,23.5]);}
        translate([7,0,5]){
            cube([50,70,20]);}

//    }
}
}
// probably radius should be 180
translate([1.5,1.5,0]){
translate([129*thou2mm,805*thou2mm]) {
cube([4.5,4.5,50]);
}
translate([129*thou2mm,1605*thou2mm]){
cube([4.5,4.5,50]);
}
translate([2292*thou2mm,805*thou2mm]){ 
    cube([4.5,4.5,50]);
}
translate([2292*thou2mm,1605*thou2mm]){ 
cube([4.5,4.5,50]);

}
}
}