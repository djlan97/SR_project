;; Auto-generated. Do not edit!


(when (boundp 'custom_msg::custom)
  (if (not (find-package "CUSTOM_MSG"))
    (make-package "CUSTOM_MSG"))
  (shadow 'custom (find-package "CUSTOM_MSG")))
(unless (find-package "CUSTOM_MSG::CUSTOM")
  (make-package "CUSTOM_MSG::CUSTOM"))

(in-package "ROS")
;;//! \htmlinclude custom.msg.html


(defclass custom_msg::custom
  :super ros::object
  :slots (_x1 _y1 _a1 _c1 _x2 _y2 _a2 _c2 _x3 _y3 _a3 _c3 _x4 _y4 _a4 _c4 ))

(defmethod custom_msg::custom
  (:init
   (&key
    ((:x1 __x1) 0.0)
    ((:y1 __y1) 0.0)
    ((:a1 __a1) 0.0)
    ((:c1 __c1) 0)
    ((:x2 __x2) 0.0)
    ((:y2 __y2) 0.0)
    ((:a2 __a2) 0.0)
    ((:c2 __c2) 0)
    ((:x3 __x3) 0.0)
    ((:y3 __y3) 0.0)
    ((:a3 __a3) 0.0)
    ((:c3 __c3) 0)
    ((:x4 __x4) 0.0)
    ((:y4 __y4) 0.0)
    ((:a4 __a4) 0.0)
    ((:c4 __c4) 0)
    )
   (send-super :init)
   (setq _x1 (float __x1))
   (setq _y1 (float __y1))
   (setq _a1 (float __a1))
   (setq _c1 (round __c1))
   (setq _x2 (float __x2))
   (setq _y2 (float __y2))
   (setq _a2 (float __a2))
   (setq _c2 (round __c2))
   (setq _x3 (float __x3))
   (setq _y3 (float __y3))
   (setq _a3 (float __a3))
   (setq _c3 (round __c3))
   (setq _x4 (float __x4))
   (setq _y4 (float __y4))
   (setq _a4 (float __a4))
   (setq _c4 (round __c4))
   self)
  (:x1
   (&optional __x1)
   (if __x1 (setq _x1 __x1)) _x1)
  (:y1
   (&optional __y1)
   (if __y1 (setq _y1 __y1)) _y1)
  (:a1
   (&optional __a1)
   (if __a1 (setq _a1 __a1)) _a1)
  (:c1
   (&optional __c1)
   (if __c1 (setq _c1 __c1)) _c1)
  (:x2
   (&optional __x2)
   (if __x2 (setq _x2 __x2)) _x2)
  (:y2
   (&optional __y2)
   (if __y2 (setq _y2 __y2)) _y2)
  (:a2
   (&optional __a2)
   (if __a2 (setq _a2 __a2)) _a2)
  (:c2
   (&optional __c2)
   (if __c2 (setq _c2 __c2)) _c2)
  (:x3
   (&optional __x3)
   (if __x3 (setq _x3 __x3)) _x3)
  (:y3
   (&optional __y3)
   (if __y3 (setq _y3 __y3)) _y3)
  (:a3
   (&optional __a3)
   (if __a3 (setq _a3 __a3)) _a3)
  (:c3
   (&optional __c3)
   (if __c3 (setq _c3 __c3)) _c3)
  (:x4
   (&optional __x4)
   (if __x4 (setq _x4 __x4)) _x4)
  (:y4
   (&optional __y4)
   (if __y4 (setq _y4 __y4)) _y4)
  (:a4
   (&optional __a4)
   (if __a4 (setq _a4 __a4)) _a4)
  (:c4
   (&optional __c4)
   (if __c4 (setq _c4 __c4)) _c4)
  (:serialization-length
   ()
   (+
    ;; float32 _x1
    4
    ;; float32 _y1
    4
    ;; float32 _a1
    4
    ;; int8 _c1
    1
    ;; float32 _x2
    4
    ;; float32 _y2
    4
    ;; float32 _a2
    4
    ;; int8 _c2
    1
    ;; float32 _x3
    4
    ;; float32 _y3
    4
    ;; float32 _a3
    4
    ;; int8 _c3
    1
    ;; float32 _x4
    4
    ;; float32 _y4
    4
    ;; float32 _a4
    4
    ;; int8 _c4
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _x1
       (sys::poke _x1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y1
       (sys::poke _y1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a1
       (sys::poke _a1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int8 _c1
       (write-byte _c1 s)
     ;; float32 _x2
       (sys::poke _x2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y2
       (sys::poke _y2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a2
       (sys::poke _a2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int8 _c2
       (write-byte _c2 s)
     ;; float32 _x3
       (sys::poke _x3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y3
       (sys::poke _y3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a3
       (sys::poke _a3 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int8 _c3
       (write-byte _c3 s)
     ;; float32 _x4
       (sys::poke _x4 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y4
       (sys::poke _y4 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a4
       (sys::poke _a4 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int8 _c4
       (write-byte _c4 s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _x1
     (setq _x1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y1
     (setq _y1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a1
     (setq _a1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int8 _c1
     (setq _c1 (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _c1 127) (setq _c1 (- _c1 256)))
   ;; float32 _x2
     (setq _x2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y2
     (setq _y2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a2
     (setq _a2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int8 _c2
     (setq _c2 (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _c2 127) (setq _c2 (- _c2 256)))
   ;; float32 _x3
     (setq _x3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y3
     (setq _y3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a3
     (setq _a3 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int8 _c3
     (setq _c3 (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _c3 127) (setq _c3 (- _c3 256)))
   ;; float32 _x4
     (setq _x4 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y4
     (setq _y4 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a4
     (setq _a4 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int8 _c4
     (setq _c4 (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _c4 127) (setq _c4 (- _c4 256)))
   ;;
   self)
  )

(setf (get custom_msg::custom :md5sum-) "1a0f9ebf965faa62489c7c7a11fc5994")
(setf (get custom_msg::custom :datatype-) "custom_msg/custom")
(setf (get custom_msg::custom :definition-)
      "float32 x1
float32 y1
float32 a1
int8 c1
float32 x2
float32 y2
float32 a2
int8 c2
float32 x3
float32 y3
float32 a3
int8 c3
float32 x4
float32 y4
float32 a4
int8 c4
")



(provide :custom_msg/custom "1a0f9ebf965faa62489c7c7a11fc5994")


