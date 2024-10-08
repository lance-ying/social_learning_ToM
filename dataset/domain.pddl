(define (domain doors-keys-gems)
    (:requirements :fluents :adl :typing)
    (:types 
        key gem - item
        item box door agent - physical
        physical color - object
    )
    (:predicates 
        (has ?a - agent ?i - item)
        (iscolor ?o - physical ?c - color)
        (offgrid ?i - item) 
        (locked ?d - door)
        (closed ?c - box)
        (hidden ?i - item)
        (inside ?k - key ?c - box)
    )
    (:functions 
        (xloc ?o - physical) (yloc ?o - physical) - integer
        (walls) - bit-matrix
    )

    (:action pickup
     :parameters (?a - agent ?i - item)
     :precondition
        (and (not (has ?a ?i)) (not (hidden ?i))
            (= (xloc ?a) (xloc ?i)) (= (yloc ?a) (yloc ?i)))
     :effect 
        (and (has ?a ?i) (offgrid ?i)
            (assign (xloc ?i) -1) (assign (yloc ?i) -1))
    )

    (:action open
     :parameters (?a - agent ?c - box)
     :precondition
        (and (closed ?c)
            (or (and (= (xloc ?a) (xloc ?c)) (= (yloc ?a) (yloc ?c)))
                (and (= (xloc ?a) (xloc ?c)) (= (- (yloc ?a) 1) (yloc ?c)))
                (and (= (xloc ?a) (xloc ?c)) (= (+ (yloc ?a) 1) (yloc ?c)))
                (and (= (- (xloc ?a) 1) (xloc ?c)) (= (yloc ?a) (yloc ?c)))
                (and (= (+ (xloc ?a) 1) (xloc ?c)) (= (yloc ?a) (yloc ?c)))))
     :effect
        (and (not (closed ?c)) 
            (forall (?k - key)
                (when (inside ?k ?c)
                    (and (not (hidden ?k)) (not (inside ?k ?c))))))
    )

    (:action unlock
     :parameters (?a - agent ?k - key ?d - door)
     :precondition
        (and (has ?a ?k) (locked ?d)
            (exists (?c - color) (and (iscolor ?k ?c) (iscolor ?d ?c)))
            (or (and (= (xloc ?a) (xloc ?d)) (= (- (yloc ?a) 1) (yloc ?d)))
                (and (= (xloc ?a) (xloc ?d)) (= (+ (yloc ?a) 1) (yloc ?d)))
                (and (= (- (xloc ?a) 1) (xloc ?d)) (= (yloc ?a) (yloc ?d)))
                (and (= (+ (xloc ?a) 1) (xloc ?d)) (= (yloc ?a) (yloc ?d)))))
     :effect
        (and (not (has ?a ?k)) (not (locked ?d)))
    )

    (:action up
     :parameters (?a - agent)
     :precondition
        (and (> (yloc ?a) 1))
            ; (= (get-index walls (- (yloc ?a) 1) (xloc ?a)) false)
            ; (not (exists (?d - door)
            ;     (and (locked ?d) (= (xloc ?a) (xloc ?d)) (= (- (yloc ?a) 1) (yloc ?d))))))
     :effect
        (and (decrease (yloc ?a) 1))
    )

    (:action down
     :parameters (?a - agent)
     :precondition
        (and (< (yloc ?a) (height walls)))
            ; (= (get-index walls (+ (yloc ?a) 1) (xloc ?a)) false)
            ; (not (exists (?d - door)
            ;     (and (locked ?d) (= (xloc ?a) (xloc ?d)) (= (+ (yloc ?a) 1) (yloc ?d))))))
     :effect 
        (and (increase (yloc ?a) 1))
    )

    (:action left
     :parameters (?a - agent)
     :precondition
        (> (xloc ?a) 1)
            ; (= (get-index walls (yloc ?a) (- (xloc ?a) 1)) false)
            ; (not (exists (?d - door)
            ;      (and (locked ?d) (= (yloc ?a) (yloc ?d)) (= (- (xloc ?a) 1) (xloc ?d))))))
     :effect
        (and (decrease (xloc ?a) 1))
    )

    (:action right
     :parameters (?a - agent)
     :precondition
        (and (< (xloc ?a) (width walls)) )
        ;     (= (get-index walls (yloc ?a) (+ (xloc ?a) 1)) false)
        ;     (not (exists (?d - door)
        ;          (and (locked ?d) (= (yloc ?a) (yloc ?d)) (= (+ (xloc ?a) 1) (xloc ?d))))))
     :effect
        (and (increase (xloc ?a) 1))
    )

)
