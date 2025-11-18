(define (domain doors-keys-gems)
    (:requirements :fluents :adl :typing)
    (:types 
        key gem - item
        item wizard door agent - physical
        physical color - object
    )
    (:predicates 
        (has ?a - agent ?i - item)
        (iscolor ?o - physical ?c - color)
        (offgrid ?i - item)
        (hidden ?i - item)
        (hold ?w - wizard ?k - key)
    )
    (:functions 
        (xloc ?o - physical) (yloc ?o - physical) - integer
        (walls) - bit-matrix
    )

    (:action pickup
     :parameters (?a - agent ?i - gem)
     :precondition
        (and (not (has ?a ?i)) (not (hidden ?i))
            (= (xloc ?a) (xloc ?i)) (= (yloc ?a) (yloc ?i)))
     :effect 
        (has ?a ?i)
    )

    (:action interact
     :parameters (?a - agent ?w - wizard)
     :precondition
        (or (and (= (xloc ?a) (xloc ?w)) (= (yloc ?a) (yloc ?w)))
                (and (= (xloc ?a) (xloc ?w)) (= (- (yloc ?a) 1) (yloc ?w)))
                (and (= (xloc ?a) (xloc ?w)) (= (+ (yloc ?a) 1) (yloc ?w)))
                (and (= (- (xloc ?a) 1) (xloc ?w)) (= (yloc ?a) (yloc ?w)))
                (and (= (+ (xloc ?a) 1) (xloc ?w)) (= (yloc ?a) (yloc ?w))))
     :effect
            (forall (?k - key)
                (when (hold ?w ?k)
                    (has ?a ?k)))
    )

    (:action pass
     :parameters (?a - agent ?k - key ?d - door)
     :precondition
        (and (has ?a ?k)
            (exists (?c - color) (and (iscolor ?k ?c) (iscolor ?d ?c)))
            (or (and (= (xloc ?a) (xloc ?d)) (= (- (yloc ?a) 1) (yloc ?d)))
                (and (= (xloc ?a) (xloc ?d)) (= (+ (yloc ?a) 1) (yloc ?d)))
                (and (= (- (xloc ?a) 1) (xloc ?d)) (= (yloc ?a) (yloc ?d)))
                (and (= (+ (xloc ?a) 1) (xloc ?d)) (= (yloc ?a) (yloc ?d)))))
     :effect
        (and (assign (xloc ?a) (xloc ?d)) (assign (yloc ?a) (yloc ?d)))
    )
    

    (:action up
     :parameters (?a - agent)
     :precondition
        (and (> (yloc ?a) 1)
            (= (get-index walls (- (yloc ?a) 1) (xloc ?a)) false)
            (not (exists (?d - door)
                (and (= (xloc ?a) (xloc ?d)) (= (- (yloc ?a) 1) (yloc ?d))))))
     :effect
        (and (decrease (yloc ?a) 1))
    )

    (:action down
     :parameters (?a - agent)
     :precondition
        (and (< (yloc ?a) (height walls))
            (= (get-index walls (+ (yloc ?a) 1) (xloc ?a)) false)
            (not (exists (?d - door)
                (and (= (xloc ?a) (xloc ?d)) (= (+ (yloc ?a) 1) (yloc ?d))))))
     :effect 
        (and (increase (yloc ?a) 1))
    )

    (:action left
     :parameters (?a - agent)
     :precondition
        (and (> (xloc ?a) 1)
            (= (get-index walls (yloc ?a) (- (xloc ?a) 1)) false)
            (not (exists (?d - door)
                 (and  (= (yloc ?a) (yloc ?d)) (= (- (xloc ?a) 1) (xloc ?d))))))
     :effect
        (and (decrease (xloc ?a) 1))
    )

    (:action right
     :parameters (?a - agent)
     :precondition
        (and (< (xloc ?a) (width walls)) 
            (= (get-index walls (yloc ?a) (+ (xloc ?a) 1)) false)
            (not (exists (?d - door)
                 (and  (= (yloc ?a) (yloc ?d)) (= (+ (xloc ?a) 1) (xloc ?d))))))
     :effect
        (and (increase (xloc ?a) 1))
    )

)
