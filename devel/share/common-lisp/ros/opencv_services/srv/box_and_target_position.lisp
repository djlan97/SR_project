; Auto-generated. Do not edit!


(cl:in-package opencv_services-srv)


;//! \htmlinclude box_and_target_position-request.msg.html

(cl:defclass <box_and_target_position-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass box_and_target_position-request (<box_and_target_position-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <box_and_target_position-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'box_and_target_position-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name opencv_services-srv:<box_and_target_position-request> is deprecated: use opencv_services-srv:box_and_target_position-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <box_and_target_position-request>) ostream)
  "Serializes a message object of type '<box_and_target_position-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <box_and_target_position-request>) istream)
  "Deserializes a message object of type '<box_and_target_position-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<box_and_target_position-request>)))
  "Returns string type for a service object of type '<box_and_target_position-request>"
  "opencv_services/box_and_target_positionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'box_and_target_position-request)))
  "Returns string type for a service object of type 'box_and_target_position-request"
  "opencv_services/box_and_target_positionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<box_and_target_position-request>)))
  "Returns md5sum for a message object of type '<box_and_target_position-request>"
  "305b8cdaf6eafb69d6dc3217db2db095")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'box_and_target_position-request)))
  "Returns md5sum for a message object of type 'box_and_target_position-request"
  "305b8cdaf6eafb69d6dc3217db2db095")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<box_and_target_position-request>)))
  "Returns full string definition for message of type '<box_and_target_position-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'box_and_target_position-request)))
  "Returns full string definition for message of type 'box_and_target_position-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <box_and_target_position-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <box_and_target_position-request>))
  "Converts a ROS message object to a list"
  (cl:list 'box_and_target_position-request
))
;//! \htmlinclude box_and_target_position-response.msg.html

(cl:defclass <box_and_target_position-response> (roslisp-msg-protocol:ros-message)
  ((x1
    :reader x1
    :initarg :x1
    :type cl:float
    :initform 0.0)
   (y1
    :reader y1
    :initarg :y1
    :type cl:float
    :initform 0.0)
   (a1
    :reader a1
    :initarg :a1
    :type cl:float
    :initform 0.0)
   (c1
    :reader c1
    :initarg :c1
    :type cl:fixnum
    :initform 0)
   (x2
    :reader x2
    :initarg :x2
    :type cl:float
    :initform 0.0)
   (y2
    :reader y2
    :initarg :y2
    :type cl:float
    :initform 0.0)
   (a2
    :reader a2
    :initarg :a2
    :type cl:float
    :initform 0.0)
   (c2
    :reader c2
    :initarg :c2
    :type cl:fixnum
    :initform 0)
   (x3
    :reader x3
    :initarg :x3
    :type cl:float
    :initform 0.0)
   (y3
    :reader y3
    :initarg :y3
    :type cl:float
    :initform 0.0)
   (a3
    :reader a3
    :initarg :a3
    :type cl:float
    :initform 0.0)
   (c3
    :reader c3
    :initarg :c3
    :type cl:fixnum
    :initform 0)
   (x4
    :reader x4
    :initarg :x4
    :type cl:float
    :initform 0.0)
   (y4
    :reader y4
    :initarg :y4
    :type cl:float
    :initform 0.0)
   (a4
    :reader a4
    :initarg :a4
    :type cl:float
    :initform 0.0)
   (c4
    :reader c4
    :initarg :c4
    :type cl:fixnum
    :initform 0))
)

(cl:defclass box_and_target_position-response (<box_and_target_position-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <box_and_target_position-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'box_and_target_position-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name opencv_services-srv:<box_and_target_position-response> is deprecated: use opencv_services-srv:box_and_target_position-response instead.")))

(cl:ensure-generic-function 'x1-val :lambda-list '(m))
(cl:defmethod x1-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:x1-val is deprecated.  Use opencv_services-srv:x1 instead.")
  (x1 m))

(cl:ensure-generic-function 'y1-val :lambda-list '(m))
(cl:defmethod y1-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:y1-val is deprecated.  Use opencv_services-srv:y1 instead.")
  (y1 m))

(cl:ensure-generic-function 'a1-val :lambda-list '(m))
(cl:defmethod a1-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:a1-val is deprecated.  Use opencv_services-srv:a1 instead.")
  (a1 m))

(cl:ensure-generic-function 'c1-val :lambda-list '(m))
(cl:defmethod c1-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:c1-val is deprecated.  Use opencv_services-srv:c1 instead.")
  (c1 m))

(cl:ensure-generic-function 'x2-val :lambda-list '(m))
(cl:defmethod x2-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:x2-val is deprecated.  Use opencv_services-srv:x2 instead.")
  (x2 m))

(cl:ensure-generic-function 'y2-val :lambda-list '(m))
(cl:defmethod y2-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:y2-val is deprecated.  Use opencv_services-srv:y2 instead.")
  (y2 m))

(cl:ensure-generic-function 'a2-val :lambda-list '(m))
(cl:defmethod a2-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:a2-val is deprecated.  Use opencv_services-srv:a2 instead.")
  (a2 m))

(cl:ensure-generic-function 'c2-val :lambda-list '(m))
(cl:defmethod c2-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:c2-val is deprecated.  Use opencv_services-srv:c2 instead.")
  (c2 m))

(cl:ensure-generic-function 'x3-val :lambda-list '(m))
(cl:defmethod x3-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:x3-val is deprecated.  Use opencv_services-srv:x3 instead.")
  (x3 m))

(cl:ensure-generic-function 'y3-val :lambda-list '(m))
(cl:defmethod y3-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:y3-val is deprecated.  Use opencv_services-srv:y3 instead.")
  (y3 m))

(cl:ensure-generic-function 'a3-val :lambda-list '(m))
(cl:defmethod a3-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:a3-val is deprecated.  Use opencv_services-srv:a3 instead.")
  (a3 m))

(cl:ensure-generic-function 'c3-val :lambda-list '(m))
(cl:defmethod c3-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:c3-val is deprecated.  Use opencv_services-srv:c3 instead.")
  (c3 m))

(cl:ensure-generic-function 'x4-val :lambda-list '(m))
(cl:defmethod x4-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:x4-val is deprecated.  Use opencv_services-srv:x4 instead.")
  (x4 m))

(cl:ensure-generic-function 'y4-val :lambda-list '(m))
(cl:defmethod y4-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:y4-val is deprecated.  Use opencv_services-srv:y4 instead.")
  (y4 m))

(cl:ensure-generic-function 'a4-val :lambda-list '(m))
(cl:defmethod a4-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:a4-val is deprecated.  Use opencv_services-srv:a4 instead.")
  (a4 m))

(cl:ensure-generic-function 'c4-val :lambda-list '(m))
(cl:defmethod c4-val ((m <box_and_target_position-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader opencv_services-srv:c4-val is deprecated.  Use opencv_services-srv:c4 instead.")
  (c4 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <box_and_target_position-response>) ostream)
  "Serializes a message object of type '<box_and_target_position-response>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'x1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'y1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'c1)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'x2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'y2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'c2)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'x3))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'y3))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a3))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'c3)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'x4))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'y4))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a4))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'c4)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <box_and_target_position-response>) istream)
  "Deserializes a message object of type '<box_and_target_position-response>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x1) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y1) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a1) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'c1)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x2) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y2) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a2) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'c2)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x3) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y3) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a3) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'c3)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x4) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y4) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a4) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'c4)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<box_and_target_position-response>)))
  "Returns string type for a service object of type '<box_and_target_position-response>"
  "opencv_services/box_and_target_positionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'box_and_target_position-response)))
  "Returns string type for a service object of type 'box_and_target_position-response"
  "opencv_services/box_and_target_positionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<box_and_target_position-response>)))
  "Returns md5sum for a message object of type '<box_and_target_position-response>"
  "305b8cdaf6eafb69d6dc3217db2db095")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'box_and_target_position-response)))
  "Returns md5sum for a message object of type 'box_and_target_position-response"
  "305b8cdaf6eafb69d6dc3217db2db095")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<box_and_target_position-response>)))
  "Returns full string definition for message of type '<box_and_target_position-response>"
  (cl:format cl:nil "float32 x1~%float32 y1~%float32 a1~%uint8 c1~%float32 x2~%float32 y2~%float32 a2~%uint8 c2~%float32 x3~%float32 y3~%float32 a3~%uint8 c3~%float32 x4~%float32 y4~%float32 a4~%uint8 c4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'box_and_target_position-response)))
  "Returns full string definition for message of type 'box_and_target_position-response"
  (cl:format cl:nil "float32 x1~%float32 y1~%float32 a1~%uint8 c1~%float32 x2~%float32 y2~%float32 a2~%uint8 c2~%float32 x3~%float32 y3~%float32 a3~%uint8 c3~%float32 x4~%float32 y4~%float32 a4~%uint8 c4~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <box_and_target_position-response>))
  (cl:+ 0
     4
     4
     4
     1
     4
     4
     4
     1
     4
     4
     4
     1
     4
     4
     4
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <box_and_target_position-response>))
  "Converts a ROS message object to a list"
  (cl:list 'box_and_target_position-response
    (cl:cons ':x1 (x1 msg))
    (cl:cons ':y1 (y1 msg))
    (cl:cons ':a1 (a1 msg))
    (cl:cons ':c1 (c1 msg))
    (cl:cons ':x2 (x2 msg))
    (cl:cons ':y2 (y2 msg))
    (cl:cons ':a2 (a2 msg))
    (cl:cons ':c2 (c2 msg))
    (cl:cons ':x3 (x3 msg))
    (cl:cons ':y3 (y3 msg))
    (cl:cons ':a3 (a3 msg))
    (cl:cons ':c3 (c3 msg))
    (cl:cons ':x4 (x4 msg))
    (cl:cons ':y4 (y4 msg))
    (cl:cons ':a4 (a4 msg))
    (cl:cons ':c4 (c4 msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'box_and_target_position)))
  'box_and_target_position-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'box_and_target_position)))
  'box_and_target_position-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'box_and_target_position)))
  "Returns string type for a service object of type '<box_and_target_position>"
  "opencv_services/box_and_target_position")