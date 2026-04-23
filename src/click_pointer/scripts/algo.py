def perform(startpoint,endpoint):
    int dx = endpoint.x-startpoint.x
    int dy = endpoint.y-startpoint.y
    int dz = endpoint.z-startpoint.z


    int curree = getee()
    #Lifting Phase 1 (adjusting y axis)
    while(abs(dy)>0.05){
        if(dy > 0){
            moveJ2up()
        }else{
            moveJ2down()
        }
        dy = endpoint.y-currpoint.y
    }

    #Moving left or right Phase 2(adjusting x axis)
    while(abs(dx)>0.05){
        if(dx < 0){
            moveJ3left()
        }else{
            moveJ3right()
        }
        dx = endpoint.x-currpoint.x
    }

    #Moving front but when J1 moves we will have to move J5 also accordingly for obvious reasons
    # and dz can't be negative since this is a feeding task,
    while(dz>0.05){
        moveJ1left()
        moveJ5littlelef()
    }


    print("fed the patient")
