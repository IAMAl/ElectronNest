; ModuleID = 'agu_code'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
@c = external global [32 x [32 x i32]], align 16

define void @c_agu() #0 {
entry:
  %i1_ptr = alloca i32, align 4
  %i2_ptr = alloca i32, align 4
  %i3_ptr = alloca i32, align 4
  store i32 0, i32* %i1_ptr, align 4
  br label %5

5:
  store i32 0, i32* %i2_ptr, align 4
  %i1 = load i32, i32* %i1_ptr, align 4
  %cond_i1 = icmp slt i32 %i1, 32
  br i1 %cond_i1, label %8, label %ret

8:
  br label %9

54:
  %i1_next = add i32 %i1, 1
  store i32 %i1_next, i32* %i1_ptr, align 4
  br label %ret

9:
  store i32 0, i32* %i3_ptr, align 4
  %i2 = load i32, i32* %i2_ptr, align 4
  %cond_i2 = icmp slt i32 %i2, 32
  br i1 %cond_i2, label %12, label %50

12:
  %sext_0_c_12 = sext i32 %i2 to i64
  %sext_1_c_12 = sext i32 %i1 to i64
  %gep_c_12 = getelementptr inbounds [32 x [32 x i32]], [32 x [32 x i32]]* @c, i64 0, i64 %sext_0_c_12, i64 %sext_1_c_12
  store i32 0, i32* %gep_c_12, align 4
  br label %19

50:
  %i2_next = add i32 %i2, 1
  store i32 %i2_next, i32* %i2_ptr, align 4
  br label %54

19:
  %i3 = load i32, i32* %i3_ptr, align 4
  %cond_i3 = icmp slt i32 %i3, 32
  br i1 %cond_i3, label %22, label %46

22:
  %sext_0_c_22 = sext i32 %i2 to i64
  %sext_1_c_22 = sext i32 %i1 to i64
  %gep_c_22 = getelementptr inbounds [32 x [32 x i32]], [32 x [32 x i32]]* @c, i64 0, i64 %sext_0_c_22, i64 %sext_1_c_22
  %load_c_22 = load i32, i32* %gep_c_22, align 4
  store i32 %45, i32* %gep_c_22, align 4
  br label %46

46:
  %i3_next = add i32 %i3, 1
  store i32 %i3_next, i32* %i3_ptr, align 4
  br label %50

ret:
  ret void
}

attributes #0 = { nounwind }