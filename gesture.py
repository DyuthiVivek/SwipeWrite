from math import dist

def coordinate(results, landmark, num):
    return float(str(results.multi_hand_landmarks[-1].landmark[landmark]).split('\n')[num].split(" ")[1])

def is_closed(results):
    if results.multi_hand_landmarks is not None:
        try:
            p0x, p0y = coordinate(results, 0, 0), coordinate(results, 0, 1)

            p7x, p7y = coordinate(results, 7, 0), coordinate(results, 7, 1)
            d07 = dist([p0x, p0y], [p7x, p7y])

            p8x, p8y = coordinate(results, 8, 0), coordinate(results, 8, 1)
            d08 = dist([p0x, p0y], [p8x, p8y])

            p11x, p11y = coordinate(results, 11, 0), coordinate(results, 11, 1)
            d011 = dist([p0x, p0y], [p11x, p11y])

            p12x, p12y = coordinate(results, 12, 0), coordinate(results, 12, 1)
            d012 = dist([p0x, p0y], [p12x, p12y])

            p15x, p15y = coordinate(results, 15, 0), coordinate(results, 15, 1)
            d015 = dist([p0x, p0y], [p15x, p15y])

            p16x, p16y = coordinate(results, 16, 0), coordinate(results, 16, 1)
            d016 = dist([p0x, p0y], [p16x, p16y])

            p19x, p19y = coordinate(results, 19, 0), coordinate(results, 19, 1)
            d019 = dist([p0x, p0y], [p19x, p19y])

            p20x, p20y = coordinate(results, 20, 0), coordinate(results, 20, 1)
            d020 = dist([p0x, p0y], [p20x, p20y])

            return d07 > d08 and d011 > d012 and d015 > d016 and d019 > d020
                    
        except:
           pass

