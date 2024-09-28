<template>
<el-button
    style="margin-left: 10px;" size="small" type="success" @click="submitUpload">开始检测
</el-button>
</template>

<script>
import t1 from "@/components/t1.vue";
import t2 from "@/components/t2.vue";
export default {
  components:{
    t1, t2
  },
  methods: {
    submitUpload() {
      this.$refs.upload.sunmit("t1");
      this.$refs.upload.submit("t2");
    }
  }
}
</script>
