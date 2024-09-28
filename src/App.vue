<template>
<div>

  <el-row>
    <el-col :span="24" class="el-header">UnetCD</el-col>
  </el-row>

  <el-row>
    <el-col :span="24" class="block"></el-col>
  </el-row>

  <el-row>
    <el-col :span="2" class="line"></el-col>
    <el-col :span="9" class="t"><t1 ref="t1"></t1></el-col>
    <el-col :span="2" class="line"></el-col>
    <el-col :span="9" class="t"><t2 ref="t2"  @transmit="getMessage"></t2></el-col>
    <el-col :span="2" class="line"></el-col>
  </el-row>

  <el-row>
    <el-col :span="24" class="block"></el-col>
  </el-row>

  <el-row>
    <el-col :span="24" class="detect">
      <el-button
          style="margin-left: 10px;" size="small" type="success" @click="submitUpload">开始检测
      </el-button></el-col>
  </el-row>

  <el-row>
    <el-col :span="2" class="line"></el-col>
    <el-col :span="12"><div class="demo-image__preview">检测结果：
      <el-image
          style="width: 150px; height: 150px; text-align: center;"
          :src="url"
          :preview-src-list= [url]
          v-if=url>
      </el-image>
    </div></el-col>
    <el-col :span="10" class="line"></el-col>
  </el-row>

</div>
</template>

<style>
.el-header{
  background-color: #409EFF;
  color: #ffffff;
  font-size: 30px;
  text-align: center;
  line-height: 80px;
}
.block{
  background-color: #ffffff;
  height: 50px;
}
.line{
  background-color: #ffffff;
  height: 10px;
}
.t{
  background-color: #ffffff;
  text-align: center;
}
.detect{
  background-color: #ffffff;
  text-align: center;
  height: 80px;
}
</style>

<script>
import t1 from "@/components/t1.vue";
import t2 from "@/components/t2.vue";

  export default {
    components: {
    t1, t2
    },
    methods:{
      submitUpload() {
        this.$refs.t1.submit();
        this.$refs.t2.submit();
      },
      getMessage(data) {
        this.url = data; //data即为子组件发送的数据
        console.log("1111")
        console.log(data)
      }
    },
    data() {
      return {
        url: ""
      };
    }
  }
</script>